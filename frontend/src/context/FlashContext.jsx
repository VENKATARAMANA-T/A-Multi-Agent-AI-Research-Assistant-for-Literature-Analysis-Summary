import { createContext, useCallback, useContext, useMemo, useRef, useState } from 'react';

const FlashContext = createContext(null);

// Long enough to read a sentence, short enough not to linger.
const AUTO_CLOSE_MS = 3000;

/**
 * Transient messages, shown top-right.
 *
 * Successes and information dismiss themselves after three seconds; errors do
 * not. An error that vanishes while you are still reading it is worse than no
 * message at all — and the one case where you may want to copy the text is the
 * case where it disappeared.
 */
export function FlashProvider({ children }) {
  const [messages, setMessages] = useState([]);
  const timers = useRef(new Map());
  const nextId = useRef(1);

  const dismiss = useCallback((id) => {
    const timer = timers.current.get(id);
    if (timer) {
      clearTimeout(timer);
      timers.current.delete(id);
    }
    setMessages((current) => current.filter((message) => message.id !== id));
  }, []);

  const push = useCallback(
    (text, tone = 'info', options = {}) => {
      if (!text) return null;
      const id = nextId.current++;
      const sticky = options.sticky ?? tone === 'error';

      setMessages((current) => [...current, { id, text: String(text), tone, sticky }]);

      if (!sticky) {
        timers.current.set(
          id,
          setTimeout(() => dismiss(id), options.duration ?? AUTO_CLOSE_MS),
        );
      }
      return id;
    },
    [dismiss],
  );

  const value = useMemo(
    () => ({
      push,
      dismiss,
      success: (text, options) => push(text, 'success', options),
      error: (text, options) => push(text, 'error', options),
      info: (text, options) => push(text, 'info', options),
      warning: (text, options) => push(text, 'warning', options),
    }),
    [push, dismiss],
  );

  return (
    <FlashContext.Provider value={value}>
      {children}
      <div className="flash-stack" role="status" aria-live="polite">
        {messages.map((message) => (
          <div key={message.id} className={`flash flash-${message.tone}`}>
            <span className="flash-icon" aria-hidden="true">
              {message.tone === 'success' ? '✓' : message.tone === 'error' ? '!' : 'i'}
            </span>
            <p className="flash-text">{message.text}</p>
            <button
              type="button"
              className="flash-close"
              aria-label="Dismiss"
              onClick={() => dismiss(message.id)}
            >
              ×
            </button>
            {!message.sticky && <span className="flash-timer" />}
          </div>
        ))}
      </div>
    </FlashContext.Provider>
  );
}

export function useFlash() {
  const context = useContext(FlashContext);
  if (!context) throw new Error('useFlash must be used inside a FlashProvider');
  return context;
}
