import { useEffect, useRef, useState } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import api from '../api/client';
import { useFlash } from '../context/FlashContext';

export default function Activate() {
  const [params] = useSearchParams();
  const flash = useFlash();
  const navigate = useNavigate();

  const [state, setState] = useState('working');
  const [message, setMessage] = useState('');
  // React 18 mounts effects twice in development; without this the token is
  // redeemed twice and the second attempt reports a confusing failure.
  const attempted = useRef(false);

  useEffect(() => {
    const token = params.get('token');
    if (attempted.current) return;
    attempted.current = true;

    if (!token) {
      setState('failed');
      setMessage('This link is missing its token.');
      return;
    }

    (async () => {
      try {
        // Called directly rather than through the auth context on purpose. The
        // endpoint hands back an access token, and adopting it would sign the
        // user in — which then bounces them off /signin, because an already
        // authenticated visitor has no business on the sign-in page. Activation
        // should prove the email address; signing in is what proves the
        // password, and that is a separate step.
        const payload = await api.activate(token);
        setState('done');
        setMessage(`Your account is active, ${payload.user.first_name || payload.user.username}.`);
        flash.success('Account activated. Sign in to continue.');
        setTimeout(() => navigate('/signin', { replace: true }), 2200);
      } catch (err) {
        setState('failed');
        setMessage(err.message || 'This link could not be used.');
        flash.error(err.message || 'Activation failed.');
      }
    })();
  }, [params, flash, navigate]);

  return (
    <div className="auth-page">
      <Link className="auth-home" to="/">
        ← ResearchCompass
      </Link>

      <div className="auth-card">
        <header className="auth-head">
          {state === 'working' && (
            <>
              <span className="auth-spinner" aria-hidden="true" />
              <h1>Activating…</h1>
              <p>One moment.</p>
            </>
          )}

          {state === 'done' && (
            <>
              <span className="auth-tick" aria-hidden="true">
                ✓
              </span>
              <h1>You are all set</h1>
              <p>{message} Taking you to the sign-in page…</p>
            </>
          )}

          {state === 'failed' && (
            <>
              <span className="auth-cross" aria-hidden="true">
                !
              </span>
              <h1>That link did not work</h1>
              <p>{message}</p>
            </>
          )}
        </header>

        {state === 'failed' && (
          <div className="callout">
            <p className="muted">
              Activation links are deliberately short-lived. Sign in with your new account and
              you will be offered a fresh one.
            </p>
            <Link className="btn btn-primary btn-sm" to="/signin">
              Go to sign in
            </Link>
          </div>
        )}

        {state === 'done' && (
          <Link className="btn btn-primary btn-block" to="/signin">
            Sign in now
          </Link>
        )}
      </div>
    </div>
  );
}
