import React, { useEffect, useRef, useState } from "react";
import { copilotChat, copilotHistory, resetCopilot } from "../api";

const SUGGESTIONS = [
  "Which model should I choose for tabular classification?",
  "What does ROC-AUC mean in simple terms?",
  "Why is BERT better than traditional NLP approaches?",
  "How do I handle class imbalance?",
  "Should I use LSTM or GRU for time series?",
  "When should I use Deep Learning vs Classic ML?",
  "What is SHAP and how do I read it?",
  "How many epochs should I train for?",
];

const WELCOME_MESSAGE = {
  role: "assistant",
  content:
    "Hi. I'm your AI Co-Pilot. I can explain results, suggest improvements, and answer ML, DL, or NLP questions. What would you like to know?",
};

function MessageBubble({ msg }) {
  const isUser = msg.role === "user";

  return (
    <div className={`mb-3 flex ${isUser ? "justify-end" : "justify-start"}`}>
      {!isUser ? (
        <div className="mt-0.5 mr-2 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-[var(--border)] bg-[var(--surface-alt)] text-[10px] font-semibold text-[var(--muted)]">
          AI
        </div>
      ) : null}

      <div
        className={`max-w-[85%] rounded-2xl px-3.5 py-2.5 text-sm leading-relaxed ${
          isUser
            ? "rounded-tr-sm bg-[var(--ink)] text-[var(--surface-solid)]"
            : "rounded-tl-sm border border-[var(--border)] bg-[var(--surface-alt)] text-[var(--ink)]"
        }`}
      >
        {msg.content.split("\n").map((line, index, lines) => (
          <React.Fragment key={`${msg.role}-${index}`}>
            {line}
            {index < lines.length - 1 ? <br /> : null}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}

export default function CoPilotPanel({ sessionId, triggerMode = "floating" }) {
  const [messages, setMessages] = useState([WELCOME_MESSAGE]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [open, setOpen] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(true);
  const bottomRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    if (!sessionId) return undefined;

    let active = true;

    async function loadHistory() {
      try {
        const response = await copilotHistory(sessionId);
        if (!active) return;

        const history = Array.isArray(response.data) ? response.data : [];
        setMessages(history.length ? history : [WELCOME_MESSAGE]);
        setShowSuggestions(history.length === 0);
      } catch {
        if (!active) return;
        setMessages([WELCOME_MESSAGE]);
        setShowSuggestions(true);
      }
    }

    setInput("");
    setMessages([WELCOME_MESSAGE]);
    setShowSuggestions(true);
    setOpen(false);
    loadHistory();

    return () => {
      active = false;
    };
  }, [sessionId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading, open]);

  useEffect(() => {
    if (!open) return undefined;

    inputRef.current?.focus();

    const onKeyDown = (event) => {
      if (event.key === "Escape") {
        setOpen(false);
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open]);

  const sendMessage = async (text) => {
    const msg = (text || input).trim();
    if (!msg || loading) return;

    setInput("");
    setShowSuggestions(false);
    setMessages((current) => [...current, { role: "user", content: msg }]);
    setLoading(true);

    try {
      const response = await copilotChat(sessionId, msg);
      setMessages((current) => [...current, { role: "assistant", content: response.data.reply }]);
    } catch {
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: "Sorry, I couldn't connect right now. Please try again.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = async () => {
    try {
      await resetCopilot(sessionId);
    } catch {
      // Ignore reset errors and still refresh the local thread.
    }

    setMessages([WELCOME_MESSAGE]);
    setInput("");
    setShowSuggestions(true);
  };

  if (!sessionId) {
    return null;
  }

  const triggerClassName =
    triggerMode === "inline"
      ? "inline-flex items-center gap-2 rounded-xl border border-[var(--border)] bg-[var(--surface-solid)] px-3 py-2 text-xs font-semibold text-[var(--muted)] hover:text-[var(--ink)]"
      : "fixed bottom-5 right-5 z-40 inline-flex items-center gap-2 rounded-full border border-[var(--border)] bg-[var(--surface-solid)] px-3 py-2 text-sm font-semibold text-[var(--ink)] shadow-[0_10px_30px_rgba(0,0,0,0.14)]";

  return (
    <>
      {!open ? (
        <button
          type="button"
          onClick={() => setOpen(true)}
          className={triggerClassName}
          aria-label="Open AI Co-Pilot"
        >
          <span className="flex h-7 w-7 items-center justify-center rounded-full border border-[var(--border)] bg-[var(--surface-alt)] text-[10px] font-semibold text-[var(--muted)]">
            AI
          </span>
          <span>Co-Pilot</span>
        </button>
      ) : null}

      {open ? (
        <>
          <button
            type="button"
            onClick={() => setOpen(false)}
            className="fixed inset-0 z-40 bg-black/20 backdrop-blur-[1px]"
            aria-label="Close AI Co-Pilot backdrop"
          />

          <aside className="fixed inset-x-4 bottom-4 top-24 z-50 flex w-auto flex-col overflow-hidden rounded-[22px] border border-[var(--border)] bg-[var(--surface-solid)] shadow-[0_20px_50px_rgba(0,0,0,0.22)] sm:left-auto sm:w-[360px]">
            <div className="flex items-center gap-3 border-b border-[var(--border)] bg-[var(--surface-alt)] px-4 py-3">
              <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full border border-[var(--border)] bg-[var(--surface-solid)] text-xs font-semibold text-[var(--ink)]">
                AI
              </div>

              <div className="min-w-0 flex-1">
                <div className="text-sm font-semibold text-[var(--ink)]">AI Co-Pilot</div>
                <div className="text-[11px] text-[var(--muted)]">Ask about models, metrics, and next steps.</div>
              </div>

              <button
                type="button"
                onClick={handleReset}
                className="rounded-lg border border-[var(--border)] bg-[var(--surface-solid)] px-2.5 py-1.5 text-[11px] font-semibold text-[var(--muted)] hover:text-[var(--ink)]"
              >
                Reset
              </button>

              <button
                type="button"
                onClick={() => setOpen(false)}
                className="inline-flex h-8 w-8 items-center justify-center rounded-lg border border-[var(--border)] bg-[var(--surface-solid)] text-[var(--muted)] hover:text-[var(--ink)]"
                aria-label="Close AI Co-Pilot"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                  <path d="M18 6 6 18" />
                  <path d="m6 6 12 12" />
                </svg>
              </button>
            </div>

            <div className="flex-1 overflow-y-auto px-4 py-4">
              {messages.map((msg, index) => (
                <MessageBubble key={`${msg.role}-${index}-${msg.content.slice(0, 18)}`} msg={msg} />
              ))}

              {loading ? (
                <div className="mb-3 flex items-center gap-2">
                  <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-[var(--border)] bg-[var(--surface-alt)] text-[10px] font-semibold text-[var(--muted)]">
                    AI
                  </div>
                  <div className="rounded-2xl rounded-tl-sm border border-[var(--border)] bg-[var(--surface-alt)] px-4 py-3">
                    <div className="flex gap-1">
                      {[0, 1, 2].map((dot) => (
                        <span
                          key={dot}
                          className="h-1.5 w-1.5 animate-bounce rounded-full bg-[var(--muted)]"
                          style={{ animationDelay: `${dot * 0.12}s` }}
                        />
                      ))}
                    </div>
                  </div>
                </div>
              ) : null}

              <div ref={bottomRef} />
            </div>

            {showSuggestions && messages.length <= 2 ? (
              <div className="border-t border-[var(--border)] px-4 py-3">
                <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--muted)]">
                  Quick questions
                </div>
                <div className="space-y-2">
                  {SUGGESTIONS.slice(0, 4).map((suggestion) => (
                    <button
                      key={suggestion}
                      type="button"
                      onClick={() => sendMessage(suggestion)}
                      className="w-full rounded-xl border border-[var(--border)] bg-[var(--surface-alt)] px-3 py-2 text-left text-xs leading-snug text-[var(--muted)] hover:text-[var(--ink)]"
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              </div>
            ) : null}

            <div className="border-t border-[var(--border)] px-3 py-3">
              <div className="flex gap-2">
                <input
                  ref={inputRef}
                  className="flex-1 rounded-xl border border-[var(--border)] bg-[var(--surface-solid)] px-3 py-2 text-sm text-[var(--ink)] outline-none focus:border-[var(--border-strong)]"
                  placeholder="Ask anything..."
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && !event.shiftKey) {
                      event.preventDefault();
                      sendMessage();
                    }
                  }}
                  disabled={loading}
                />
                <button
                  type="button"
                  onClick={() => sendMessage()}
                  disabled={loading || !input.trim()}
                  className="inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-[var(--ink)] text-[var(--surface-solid)] disabled:opacity-40"
                  aria-label="Send message"
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                    <path d="m22 2-7 20-4-9-9-4Z" />
                    <path d="M22 2 11 13" />
                  </svg>
                </button>
              </div>
            </div>
          </aside>
        </>
      ) : null}
    </>
  );
}
