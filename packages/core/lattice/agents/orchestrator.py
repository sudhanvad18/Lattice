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
from typing import Any

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
- 2-4 subtasks is typical for most tasks"""

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
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(content[start:end])
                subtasks = []
                subtask_ids: list[str] = []

                for item in data.get("subtasks", []):
                    role_str = item.get("assigned_to", "researcher")
                    try:
                        role = AgentRole(role_str)
                    except ValueError:
                        role = AgentRole.RESEARCHER

                    st = SubTask(
                        description=item["description"],
                        assigned_to=role,
                    )
                    subtask_ids.append(st.id)
                    subtasks.append(st)

                # Resolve depends_on (by index reference)
                for i, item in enumerate(data.get("subtasks", [])):
                    deps = item.get("depends_on", [])
                    for dep_idx in deps:
                        if isinstance(dep_idx, int) and 0 <= dep_idx < len(subtask_ids):
                            subtasks[i].depends_on.append(subtask_ids[dep_idx])

                return TaskPlan(task_description=task_description, subtasks=subtasks)
        except (json.JSONDecodeError, KeyError, IndexError) as e:
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

    def build_graph(self) -> StateGraph:
        """Build the LangGraph state machine for task execution.

        Flow:
            plan → route_next → [researcher|writer] → review → route_after_review → [done|revise|next]
        """

        graph = StateGraph(dict)

        # --- Node functions ---

        async def plan_node(state_dict: dict) -> dict:
            state = AgentState(**state_dict)
            state = await self.execute(state)
            return state.model_dump()

        async def researcher_node(state_dict: dict) -> dict:
            state = AgentState(**state_dict)
            state.current_agent = AgentRole.RESEARCHER
            state = await self._researcher.execute(state)
            # Mark subtask complete
            if state.current_subtask_id and state.plan:
                for st in state.plan.subtasks:
                    if st.id == state.current_subtask_id:
                        st.status = TaskStatus.COMPLETED
                        st.result_artifact_id = (
                            state.artifacts[-1].id if state.artifacts else None
                        )
            return state.model_dump()

        async def writer_node(state_dict: dict) -> dict:
            state = AgentState(**state_dict)
            state.current_agent = AgentRole.WRITER
            state = await self._writer.execute(state)
            if state.current_subtask_id and state.plan:
                for st in state.plan.subtasks:
                    if st.id == state.current_subtask_id:
                        st.status = TaskStatus.NEEDS_REVIEW
            return state.model_dump()

        async def reviewer_node(state_dict: dict) -> dict:
            state = AgentState(**state_dict)
            state.current_agent = AgentRole.REVIEWER
            state = await self._reviewer.execute(state)
            return state.model_dump()

        # --- Routing functions ---

        def route_after_prepare(state_dict: dict) -> str:
            """Route based on the subtask that prepare_next just selected."""
            state = AgentState(**state_dict)
            if not state.current_subtask_id or not state.plan:
                return "done"

            for st in state.plan.subtasks:
                if st.id == state.current_subtask_id:
                    if st.assigned_to == AgentRole.RESEARCHER:
                        return "researcher"
                    elif st.assigned_to == AgentRole.WRITER:
                        return "writer"
            return "done"

        def route_after_process_review(state_dict: dict) -> str:
            """After processing the review, decide next step."""
            state = AgentState(**state_dict)
            if not state.reviews:
                return "prepare_next"

            latest_review = state.reviews[-1]
            if latest_review.approved:
                return "prepare_next"
            else:
                if state.iteration_count >= state.max_iterations:
                    return "done"
                return "writer"

        def prepare_next(state_dict: dict) -> dict:
            """Set current_subtask_id before dispatching to an agent."""
            state = AgentState(**state_dict)
            next_st = state.get_next_subtask()
            if next_st:
                state.current_subtask_id = next_st.id
                next_st.status = TaskStatus.IN_PROGRESS
            else:
                state.current_subtask_id = None
            state.iteration_count += 1
            return state.model_dump()

        def process_review(state_dict: dict) -> dict:
            """Process review verdict: mark subtask complete if approved."""
            state = AgentState(**state_dict)
            if state.reviews:
                latest_review = state.reviews[-1]
                if latest_review.approved and state.current_subtask_id and state.plan:
                    for st in state.plan.subtasks:
                        if st.id == state.current_subtask_id:
                            st.status = TaskStatus.COMPLETED
            return state.model_dump()

        def finalize(state_dict: dict) -> dict:
            """Mark the overall task as complete."""
            state = AgentState(**state_dict)
            all_subtasks_done = state.plan and all(
                st.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
                for st in state.plan.subtasks
            )
            # Check only the latest review for each artifact
            latest_review_approved = (
                state.reviews[-1].approved if state.reviews else True
            )
            if all_subtasks_done:
                state.status = TaskStatus.COMPLETED
            elif state.iteration_count >= state.max_iterations:
                state.status = TaskStatus.COMPLETED
            elif latest_review_approved and not state.get_pending_subtasks():
                state.status = TaskStatus.COMPLETED
            else:
                state.status = TaskStatus.FAILED
            return state.model_dump()

        # --- Build graph ---
        graph.add_node("plan", plan_node)
        graph.add_node("prepare_next", prepare_next)
        graph.add_node("researcher", researcher_node)
        graph.add_node("writer", writer_node)
        graph.add_node("reviewer", reviewer_node)
        graph.add_node("process_review", process_review)
        graph.add_node("done", finalize)

        graph.set_entry_point("plan")
        graph.add_edge("plan", "prepare_next")

        graph.add_conditional_edges(
            "prepare_next",
            route_after_prepare,
            {
                "researcher": "researcher",
                "writer": "writer",
                "done": "done",
            },
        )

        graph.add_edge("researcher", "prepare_next")
        graph.add_edge("writer", "reviewer")
        graph.add_edge("reviewer", "process_review")

        graph.add_conditional_edges(
            "process_review",
            route_after_process_review,
            {
                "prepare_next": "prepare_next",
                "writer": "writer",
                "done": "done",
            },
        )

        graph.add_edge("done", END)

        return graph

    async def run(
        self,
        task: str,
        max_iterations: int = 10,
        checkpoint_store: "CheckpointStore | None" = None,
    ) -> AgentState:
        """Execute the full agent workflow for a task.

        This is the main entry point — give it a task and it returns
        the final state with all artifacts.

        If checkpoint_store is provided, saves state after each node
        execution and can resume from crashes.
        """
        from lattice.comms.checkpoint import CheckpointStore

        initial_state = AgentState(
            original_task=task,
            max_iterations=max_iterations,
        )

        # Try to resume from checkpoint
        if checkpoint_store:
            saved = await checkpoint_store.load(initial_state.task_id)
            if saved:
                logger.info("orchestrator_resuming", task_id=initial_state.task_id)
                initial_state = AgentState(**saved)

        graph = self.build_graph()
        compiled = graph.compile()

        result = await compiled.ainvoke(initial_state.model_dump())
        final_state = AgentState(**result)

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
