import { useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import api from '../api/client';
import { useAuth } from '../context/AuthContext';
import { useFlash } from '../context/FlashContext';

export default function SignIn() {
  const { signIn } = useAuth();
  const flash = useFlash();
  const navigate = useNavigate();
  const location = useLocation();

  const [identifier, setIdentifier] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [busy, setBusy] = useState(false);
  // Shown only when the server says the account exists but is not activated,
  // so the offer to resend appears exactly when it is useful.
  const [needsActivation, setNeedsActivation] = useState(false);

  // Where the user was headed before being asked to sign in.
  const destination = location.state?.from || '/app';

  const submit = async (event) => {
    event.preventDefault();
    if (busy) return;

    if (!identifier.trim() || !password) {
      flash.error('Enter your username or email, and your password.');
      return;
    }

    setBusy(true);
    setNeedsActivation(false);
    try {
      const user = await signIn(identifier.trim(), password);
      flash.success(`Welcome back, ${user.first_name || user.username}.`);
      navigate(destination, { replace: true });
    } catch (err) {
      if (err.status === 403) {
        setNeedsActivation(true);
        flash.error(err.message);
      } else {
        flash.error(err.message || 'Could not sign in.');
      }
      setPassword('');
    } finally {
      setBusy(false);
    }
  };

  const resend = async () => {
    try {
      const reply = await api.resendActivation(identifier.trim());
      flash.info(reply.message);
    } catch (err) {
      flash.error(err.message);
    }
  };

  return (
    <div className="auth-page">
      <Link className="auth-home" to="/">
        ← ResearchCompass
      </Link>

      <div className="auth-card">
        <header className="auth-head">
          <h1>Welcome back</h1>
          <p>Sign in to your corpus.</p>
        </header>

        <form onSubmit={submit} className="auth-form" noValidate>
          <label className="field">
            Username or email
            <input
              className="input"
              value={identifier}
              onChange={(event) => setIdentifier(event.target.value)}
              autoComplete="username"
              autoFocus
              placeholder="user1 or user1@gmail.com"
            />
          </label>

          <label className="field">
            Password
            <div className="input-with-button">
              <input
                className="input"
                type={showPassword ? 'text' : 'password'}
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                autoComplete="current-password"
              />
              <button
                type="button"
                className="input-button"
                onClick={() => setShowPassword((current) => !current)}
                aria-label={showPassword ? 'Hide password' : 'Show password'}
              >
                {showPassword ? 'Hide' : 'Show'}
              </button>
            </div>
          </label>

          <button type="submit" className="btn btn-primary btn-block" disabled={busy}>
            {busy ? 'Signing in…' : 'Sign in'}
          </button>

          {needsActivation && (
            <div className="callout">
              <strong>This account is not activated yet.</strong>
              <p className="muted">
                The link was emailed when the account was created and expires quickly.
              </p>
              <button type="button" className="btn btn-ghost btn-sm" onClick={resend}>
                Send a new activation link
              </button>
            </div>
          )}
        </form>

        <footer className="auth-foot">
          No account yet? <Link to="/signup">Create one</Link>
        </footer>
      </div>
    </div>
  );
}
