import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import api from '../api/client';
import { useFlash } from '../context/FlashContext';

const EMPTY = {
  first_name: '',
  last_name: '',
  username: '',
  email: '',
  password: '',
  confirm_password: '',
};

const STRENGTH_TONE = ['danger', 'danger', 'warning', 'success', 'success'];

export default function SignUp() {
  const flash = useFlash();

  const [form, setForm] = useState(EMPTY);
  const [fieldErrors, setFieldErrors] = useState({});
  const [strength, setStrength] = useState(null);
  const [rules, setRules] = useState({ min_length: 6, activation_minutes: 5 });
  const [busy, setBusy] = useState(false);
  const [sent, setSent] = useState(null);

  useEffect(() => {
    api.passwordRules().then(setRules).catch(() => {});
  }, []);

  // Scored as you type, but only ever advisory — the server decides what is
  // actually refused, and a weak-but-allowed password still goes through.
  useEffect(() => {
    if (!form.password) {
      setStrength(null);
      return undefined;
    }
    const timer = setTimeout(() => {
      api.passwordStrength(form.password).then(setStrength).catch(() => {});
    }, 250);
    return () => clearTimeout(timer);
  }, [form.password]);

  const update = (field) => (event) => {
    setForm((current) => ({ ...current, [field]: event.target.value }));
    setFieldErrors((current) => {
      if (!current[field]) return current;
      const next = { ...current };
      delete next[field];
      return next;
    });
  };

  const submit = async (event) => {
    event.preventDefault();
    if (busy) return;

    const local = {};
    if (!form.first_name.trim()) local.first_name = ['Enter your first name.'];
    if (!form.last_name.trim()) local.last_name = ['Enter your last name.'];
    if (form.password !== form.confirm_password) {
      local.confirm_password = ['The two passwords do not match.'];
    }
    if (form.password.length < rules.min_length) {
      local.password = [`Use at least ${rules.min_length} characters.`];
    }

    if (Object.keys(local).length) {
      setFieldErrors(local);
      flash.error('Fix the highlighted fields and try again.');
      return;
    }

    setBusy(true);
    setFieldErrors({});
    try {
      const reply = await api.register(form);
      setSent(reply);
      flash.success(reply.message);
    } catch (err) {
      // 422 carries {field: [messages]} so each input can be marked.
      const detail = err.payload?.detail;
      if (detail && typeof detail === 'object' && !Array.isArray(detail)) {
        setFieldErrors(detail);
        flash.error('Fix the highlighted fields and try again.');
      } else {
        flash.error(err.message || 'Could not create the account.');
      }
    } finally {
      setBusy(false);
    }
  };

  const errorFor = (field) =>
    fieldErrors[field] ? (
      <ul className="field-errors">
        {fieldErrors[field].map((message) => (
          <li key={message}>{message}</li>
        ))}
      </ul>
    ) : null;

  if (sent) {
    return (
      <div className="auth-page">
        <Link className="auth-home" to="/">
          ← ResearchCompass
        </Link>

        <div className="auth-card">
          <header className="auth-head">
            <span className="auth-tick" aria-hidden="true">
              ✓
            </span>
            <h1>Check your email</h1>
            <p>
              We sent a link to <strong>{form.email}</strong>. It activates your account and
              expires in {rules.activation_minutes} minutes.
            </p>
          </header>

          {!sent.email_sent && (
            <div className="callout callout-warning">
              <strong>No mail server is configured on this install.</strong>
              <p className="muted">{sent.detail}</p>
              {sent.activation_link && (
                <a className="btn btn-primary btn-sm" href={sent.activation_link}>
                  Activate my account →
                </a>
              )}
            </div>
          )}

          <footer className="auth-foot">
            Already activated? <Link to="/signin">Sign in</Link>
          </footer>
        </div>
      </div>
    );
  }

  return (
    <div className="auth-page">
      <Link className="auth-home" to="/">
        ← ResearchCompass
      </Link>

      <div className="auth-card auth-card-wide">
        <header className="auth-head">
          <h1>Create your account</h1>
          <p>Your papers, your corpus, your workspace.</p>
        </header>

        <form onSubmit={submit} className="auth-form" noValidate>
          <div className="field-row">
            <label className="field">
              First name
              <input
                className={`input ${fieldErrors.first_name ? 'is-invalid' : ''}`}
                value={form.first_name}
                onChange={update('first_name')}
                autoComplete="given-name"
                autoFocus
              />
              {errorFor('first_name')}
            </label>

            <label className="field">
              Last name
              <input
                className={`input ${fieldErrors.last_name ? 'is-invalid' : ''}`}
                value={form.last_name}
                onChange={update('last_name')}
                autoComplete="family-name"
              />
              {errorFor('last_name')}
            </label>
          </div>

          <label className="field">
            Username
            <input
              className={`input ${fieldErrors.username ? 'is-invalid' : ''}`}
              value={form.username}
              onChange={update('username')}
              autoComplete="username"
              placeholder="how you sign in"
            />
            {errorFor('username')}
          </label>

          <label className="field">
            Email
            <input
              className={`input ${fieldErrors.email ? 'is-invalid' : ''}`}
              type="email"
              value={form.email}
              onChange={update('email')}
              autoComplete="email"
              placeholder="where the activation link goes"
            />
            {errorFor('email')}
          </label>

          <div className="field-row">
            <label className="field">
              Password
              <input
                className={`input ${fieldErrors.password ? 'is-invalid' : ''}`}
                type="password"
                value={form.password}
                onChange={update('password')}
                autoComplete="new-password"
              />
              {errorFor('password')}
            </label>

            <label className="field">
              Confirm password
              <input
                className={`input ${fieldErrors.confirm_password ? 'is-invalid' : ''}`}
                type="password"
                value={form.confirm_password}
                onChange={update('confirm_password')}
                autoComplete="new-password"
              />
              {errorFor('confirm_password')}
            </label>
          </div>

          {strength && (
            <div className="strength">
              <div className={`strength-bar is-${STRENGTH_TONE[strength.score]}`}>
                <span style={{ width: `${((strength.score + 1) / 5) * 100}%` }} />
              </div>
              <div className="strength-label">
                Strength: <strong>{strength.label}</strong>
                {strength.suggestions.length > 0 && (
                  <span className="muted"> · {strength.suggestions[0]}</span>
                )}
              </div>
            </div>
          )}

          <p className="muted auth-rules">
            At least {rules.min_length} characters. Length matters more than symbols — the
            strength meter is advice, not a gate.
          </p>

          <button type="submit" className="btn btn-primary btn-block" disabled={busy}>
            {busy ? 'Creating your account…' : 'Create account'}
          </button>
        </form>

        <footer className="auth-foot">
          Already have an account? <Link to="/signin">Sign in</Link>
        </footer>
      </div>
    </div>
  );
}
