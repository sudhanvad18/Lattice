"""
Orchestrator — LangGraph state machine.

The orchestrator is the brain of the agent team. It:
1. Decomposes a high-level task into subtasks
2. Assigns subtasks to specialist agents
3. Routes artifacts between agents
4. Manages the review loop (write → review → revise if rejected)
5. Determines when the task is complete

Built on LangGraph for structured, observable multi-agent flows.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from graphlib import TopologicalSorter
from typing import TYPE_CHECKING, Any

import structlog
from langgraph.graph import END, StateGraph

from lattice.agents.base import BaseAgent
from lattice.agents.researcher import ResearcherAgent
from lattice.agents.reviewer import ReviewerAgent
from lattice.agents.state import (
    AgentRole,
    AgentState,
    SubTask,
    TaskPlan,
    TaskStatus,
)
from lattice.agents.writer import WriterAgent
from lattice.inference.provider import LLMProvider

if TYPE_CHECKING:
    from lattice.comms.checkpoint import CheckpointStore

RunObserver = Callable[[AgentState, str, dict[str, Any]], None]

logger = structlog.get_logger()


class OrchestratorAgent(BaseAgent):
    """Decomposes tasks and coordinates the agent team."""

    role = AgentRole.ORCHESTRATOR

    def __init__(
        self,
        provider: LLMProvider,
        researcher: ResearcherAgent | None = None,
        writer: WriterAgent | None = None,
        reviewer: ReviewerAgent | None = None,
        **kwargs,
    ) -> None:
        super().__init__(provider, **kwargs)
        self._researcher = researcher or ResearcherAgent(provider)
        self._writer = writer or WriterAgent(provider)
        self._reviewer = reviewer or ReviewerAgent(provider)

    @property
    def system_prompt(self) -> str:
        return """You are an Orchestrator in an autonomous agent team. Your job is to:

1. Break down a high-level task into specific, actionable subtasks
2. Assign each subtask to the appropriate specialist agent
3. Define dependencies between subtasks

Available agents:
- researcher: Queries knowledge base, gathers information, produces research briefs
- writer: Produces documentation, reports, wiki pages from research
- reviewer: Validates quality of outputs (automatically added, don't include in plan)

Your output MUST be a JSON object with this structure:
{
    "subtasks": [
        {
            "description": "what needs to be done",
            "assigned_to": "researcher|writer",
            "depends_on": []
        }
    ]
}

Rules:
- Always start with a research phase before writing
- Keep subtasks focused and specific
- Writing tasks should depend on research tasks
- The reviewer is invoked automatically after writing, don't plan it explicitly
- Prefer one KB research step followed by one briefing. Combine related research questions.
- Identify evidence gaps. Code examples are illustrative only. Never execute code or write files.
- Dependencies are zero-based indices into subtasks."""

    async def execute(self, state: AgentState) -> AgentState:
        """Create a task plan (only called during planning phase)."""
        result = await self._call_llm(
            f"Decompose this task into subtasks:\n\n{state.original_task}"
        )

        plan = self._parse_plan(result.content, state.original_task)
        state.plan = plan
        state.status = TaskStatus.IN_PROGRESS
        logger.info("orchestrator_planned", subtasks=len(plan.subtasks))
        return state

    def _parse_plan(self, content: str, task_description: str) -> TaskPlan:
        """Parse the LLM's task plan from JSON."""
        try:
            payload = content.strip()
            if payload.startswith("```") and payload.endswith("```"):
                _, _, payload = payload[:-3].partition("\n")
            data = json.loads(payload)
            items = data["subtasks"]
            if not isinstance(items, list) or not items:
                raise ValueError("Expected a nonempty subtask list")
            subtasks = []
            dependencies = {}
            for i, item in enumerate(items):
                role = AgentRole(item["assigned_to"])
                if role not in (AgentRole.RESEARCHER, AgentRole.WRITER):
                    raise ValueError("Unsupported plan role")
                description = item["description"]
                if not isinstance(description, str) or not description.strip():
                    raise ValueError("Missing subtask description")
                deps = item.get("depends_on", [])
                if not isinstance(deps, list) or any(
                    type(dep) is not int or not 0 <= dep < len(items) for dep in deps
                ):
                    raise ValueError("Invalid dependency index")
                dependencies[i] = deps
                subtasks.append(SubTask(description=description, assigned_to=role))

            TopologicalSorter(dependencies).prepare()  # Raises CycleError on cycles.
            for i, deps in dependencies.items():
                subtasks[i].depends_on = [subtasks[d].id for d in deps]
            return TaskPlan(task_description=task_description, subtasks=subtasks)
        except (ValueError, KeyError, TypeError, RecursionError) as e:
            logger.warning("orchestrator_plan_parse_failed", error=str(e))

        # Fallback: simple research → write plan
        research_st = SubTask(
            description=f"Research: {task_description}",
            assigned_to=AgentRole.RESEARCHER,
        )
        write_st = SubTask(
            description=f"Write: {task_description}",
            assigned_to=AgentRole.WRITER,
            depends_on=[research_st.id],
        )
        return TaskPlan(
            task_description=task_description,
            subtasks=[research_st, write_st],
        )

    def build_graph(self, observer: RunObserver | None = None) -> StateGraph:
        """Each dispatch consumes one iteration, including a revision."""
        graph = StateGraph(dict)

        def publish(state: AgentState, event: str, message: str, **details: Any) -> None:
            state.updated_at = datetime.now(UTC)
            if observer:
                observer(state, event, {
                    "agent": state.current_agent.value, "message": message, **details,
                })

        def start(state: AgentState, role: AgentRole, message: str) -> None:
            state.current_agent = role
            publish(state, "agent_started", message)

        async def plan_node(raw: dict) -> dict:
            state = AgentState(**raw)
            state.status = TaskStatus.IN_PROGRESS
            start(state, self.role, "Planning the task")
            state = await self.execute(state)
            count = len(state.plan.subtasks) if state.plan else 0
            publish(state, "plan_created", f"Planned {count} subtasks", subtasks=count)
            publish(state, "agent_completed", "Finished planning")
            return state.model_dump()

        def prepare(raw: dict) -> dict:
            state = AgentState(**raw)
            subtasks = state.plan.subtasks if state.plan else []
            failed = next((st for st in subtasks if st.status == TaskStatus.FAILED), None)
            if state.status == TaskStatus.FAILED:
                reason = "Agent execution failed. Check the task audit and retry."
            elif failed:
                reason = f"Required subtask failed: {failed.description}"
            elif subtasks and all(st.status == TaskStatus.COMPLETED for st in subtasks):
                state.status = TaskStatus.COMPLETED
                state.current_subtask_id = None
                return state.model_dump()
            elif state.iteration_count >= state.max_iterations:
                reason = f"Iteration limit ({state.max_iterations}) reached with unfinished work."
                if state.reviews and not state.reviews[-1].approved:
                    review = state.reviews[-1]
                    feedback = "; ".join(review.issues) or review.reasoning or "Revision required"
                    reason += f" Review rejected: {feedback}"
            else:
                next_st = next((st for st in subtasks if st.status == TaskStatus.REJECTED), None)
                next_st = next_st or state.get_next_subtask()
                if next_st and next_st.assigned_to in (AgentRole.RESEARCHER, AgentRole.WRITER):
                    next_st.status = TaskStatus.IN_PROGRESS
                    state.current_subtask_id = next_st.id
                    state.iteration_count += 1
                    return state.model_dump()
                reason = "No executable subtask. Check the plan and its dependencies."
            state.status = TaskStatus.FAILED
            state.errors.append(reason)
            return state.model_dump()

        async def dispatch(raw: dict) -> dict:
            state = AgentState(**raw)
            subtask = next(st for st in state.plan.subtasks if st.id == state.current_subtask_id)
            agent = (self._researcher if subtask.assigned_to == AgentRole.RESEARCHER
                     else self._writer)
            start(state, agent.role, subtask.description)
            before = len(state.artifacts)
            state = await agent.execute(state)
            subtask = next(st for st in state.plan.subtasks if st.id == state.current_subtask_id)
            produced = state.artifacts[before:]
            for artifact in produced:
                publish(state, "artifact_produced", f"Produced {artifact.artifact_type}",
                        artifact_id=artifact.id)
            if not produced or TaskStatus.FAILED in (state.status, subtask.status):
                subtask.status = TaskStatus.FAILED
                return state.model_dump()
            subtask.result_artifact_id = produced[-1].id
            subtask.status = (TaskStatus.NEEDS_REVIEW if agent.role == AgentRole.WRITER
                              else TaskStatus.COMPLETED)
            publish(state, "agent_completed", f"Finished {subtask.description}")
            return state.model_dump()

        async def review_node(raw: dict) -> dict:
            state = AgentState(**raw)
            start(state, AgentRole.REVIEWER, "Reviewing the draft")
            before = len(state.reviews)
            state = await self._reviewer.execute(state)
            subtask = next(st for st in state.plan.subtasks if st.id == state.current_subtask_id)
            if (len(state.reviews) == before
                    or state.reviews[-1].artifact_id != subtask.result_artifact_id):
                subtask.status = TaskStatus.FAILED
                state.errors.append("Reviewer returned no verdict for the current draft.")
                return state.model_dump()
            verdict = state.reviews[-1]
            subtask.status = TaskStatus.COMPLETED if verdict.approved else TaskStatus.REJECTED
            message = "Draft approved" if verdict.approved else "Draft rejected"
            publish(state, "review_complete", message, approved=verdict.approved,
                    reasoning=verdict.reasoning, issues=verdict.issues)
            publish(state, "agent_completed", "Finished reviewing the draft")
            return state.model_dump()

        def after_dispatch(raw: dict) -> str:
            if any(st["status"] == TaskStatus.NEEDS_REVIEW for st in raw["plan"]["subtasks"]):
                return "review"
            return "prepare"

        graph.add_node("plan", plan_node)
        graph.add_node("prepare", prepare)
        graph.add_node("dispatch", dispatch)
        graph.add_node("review", review_node)
        graph.set_entry_point("plan")
        graph.add_edge("plan", "prepare")
        graph.add_conditional_edges("prepare", lambda raw: (
            "dispatch" if raw["status"] == TaskStatus.IN_PROGRESS else END
        ))
        graph.add_conditional_edges("dispatch", after_dispatch)
        graph.add_edge("review", "prepare")
        return graph

    async def run(
        self,
        task: str,
        max_iterations: int = 10,
        checkpoint_store: CheckpointStore | None = None,
        *,
        initial_state: AgentState | None = None,
        observer: RunObserver | None = None,
    ) -> AgentState:
        """Execute the full agent workflow for a task.

        This is the main entry point — give it a task and it returns
        the final state with all artifacts.

        An optional observer receives progress during each agent phase.
        """
        initial_state = initial_state or AgentState(
            original_task=task,
            max_iterations=max_iterations,
        )

        # Try to resume from checkpoint
        if checkpoint_store:
            saved = await checkpoint_store.load(initial_state.task_id)
            if saved:
                logger.info("orchestrator_resuming", task_id=initial_state.task_id)
                initial_state = AgentState(**saved)

        graph = self.build_graph(observer)
        compiled = graph.compile()

        result = await compiled.ainvoke(
            initial_state.model_dump(),
            config={"recursion_limit": 3 * max(1, initial_state.max_iterations) + 4},
        )
        final_state = AgentState(**result)

        final_state.updated_at = datetime.now(UTC)

        # Save final state
        if checkpoint_store:
            await checkpoint_store.save(final_state.task_id, final_state.model_dump())

        logger.info(
            "orchestrator_run_complete",
            task=task[:50],
            status=final_state.status.value,
            artifacts=len(final_state.artifacts),
            iterations=final_state.iteration_count,
        )
        return final_state
