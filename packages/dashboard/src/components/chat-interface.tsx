"use client";

import { useState, useRef, useEffect } from "react";
import { Send, Loader2, BookOpen } from "lucide-react";
import { chatQuery } from "@/lib/api";

interface Message {
  role: "user" | "assistant";
  content: string;
  citations?: string[];
  timestamp: Date;
}

export function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMsg: Message = { role: "user", content: input, timestamp: new Date() };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const res = await chatQuery(input);
      const assistantMsg: Message = {
        role: "assistant",
        content: res.response,
        citations: res.citations,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch {
      const errorMsg: Message = {
        role: "assistant",
        content: "Sorry, I couldn't process that request. Make sure the Lattice API is running.",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="h-full flex flex-col p-6">
      <div className="mb-4">
        <h1 className="text-2xl font-bold">Research Chat</h1>
        <p className="text-zinc-400 text-sm mt-1">
          Ask questions — powered by the Researcher agent with RAG over your knowledge base
        </p>
      </div>

      {/* Messages */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto scrollbar-thin space-y-4 mb-4">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-zinc-600">
            <BookOpen className="w-10 h-10 mb-3" />
            <p className="text-sm">Ask anything about your knowledge base.</p>
            <div className="mt-4 grid grid-cols-2 gap-2">
              {[
                "What are the main failure modes for turbofan engines?",
                "Summarize our authentication architecture",
                "What inspection procedures exist?",
                "How does the data pipeline work?",
              ].map((q, i) => (
                <button
                  key={i}
                  onClick={() => setInput(q)}
                  className="text-xs text-left px-3 py-2 bg-zinc-900 border border-zinc-800 rounded-lg hover:border-zinc-700 text-zinc-400 hover:text-zinc-300 transition-colors"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
            <div
              className={`max-w-[75%] rounded-xl px-4 py-3 ${
                msg.role === "user"
                  ? "bg-emerald-900/30 border border-emerald-800 text-zinc-200"
                  : "bg-zinc-900 border border-zinc-800 text-zinc-300"
              }`}
            >
              <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
              {msg.citations && msg.citations.length > 0 && (
                <div className="mt-2 pt-2 border-t border-zinc-800 flex flex-wrap gap-1">
                  {msg.citations.map((c, j) => (
                    <span key={j} className="text-xs px-1.5 py-0.5 bg-zinc-800 rounded text-zinc-500">
                      {c}
                    </span>
                  ))}
                </div>
              )}
              <span className="text-xs text-zinc-600 mt-1 block">
                {msg.timestamp.toLocaleTimeString()}
              </span>
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-zinc-900 border border-zinc-800 rounded-xl px-4 py-3">
              <Loader2 className="w-4 h-4 animate-spin text-zinc-500" />
            </div>
          </div>
        )}
      </div>

      {/* Input */}
      <form onSubmit={handleSend} className="flex gap-2">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question..."
          className="flex-1 bg-zinc-900 border border-zinc-800 rounded-xl px-4 py-3 text-sm text-zinc-200 placeholder-zinc-600 outline-none focus:border-zinc-700 transition-colors"
        />
        <button
          type="submit"
          disabled={loading || !input.trim()}
          className="px-4 bg-emerald-600 hover:bg-emerald-500 disabled:bg-zinc-800 disabled:text-zinc-600 text-white rounded-xl transition-colors"
        >
          <Send className="w-4 h-4" />
        </button>
      </form>
    </div>
  );
}
