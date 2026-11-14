"use client";

import { Component, ReactNode } from "react";
import { AlertTriangle } from "lucide-react";

interface Props {
  label: string;
  children: ReactNode;
}

interface State {
  error: Error | null;
}

export class ViewErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error) {
    console.error(`${this.props.label} view crashed`, error);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="h-full flex flex-col items-center justify-center gap-3 p-6 text-sm text-zinc-400">
        <AlertTriangle className="w-6 h-6 text-amber-400" />
        <p>The {this.props.label} view hit an error.</p>
        <pre className="max-w-xl whitespace-pre-wrap text-xs text-zinc-500">{this.state.error.message}</pre>
        <button onClick={() => this.setState({ error: null })}
          className="px-3 py-1.5 bg-zinc-800 rounded-lg hover:bg-zinc-700">
          Try again
        </button>
      </div>
    );
  }
}
