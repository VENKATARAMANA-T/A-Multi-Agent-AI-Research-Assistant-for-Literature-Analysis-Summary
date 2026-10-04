import { useState } from 'react';
import api from '../api/client';
import { Badge, Card } from '../components/common';
import { useAuth } from '../context/AuthContext';
import { useCorpus } from '../context/CorpusContext';
import { useFlash } from '../context/FlashContext';

const EMPTY = { current_password: '', new_password: '', confirm_password: '' };

export default function Profile() {
  const { user, signOut } = useAuth();
  const { indexedPapers, papers } = useCorpus();
  const flash = useFlash();

  const [form, setForm] = useState(EMPTY);
  const [errors, setErrors] = useState({});
  const [busy, setBusy] = useState(false);

  const update = (field) => (event) => {
    setForm((current) => ({ ...current, [field]: event.target.value }));
    setErrors((current) => {
      if (!current[field]) return current;
      const next = { ...current };
      delete next[field];
      return next;
    });
  };

  const submit = async (event) => {
    event.preventDefault();
    if (busy) return;

    if (form.new_password !== form.confirm_password) {
      setErrors({ confirm_password: ['The two passwords do not match.'] });
      flash.error('The two passwords do not match.');
      return;
    }

    setBusy(true);
    setErrors({});
    try {
      const reply = await api.changePassword(form);
      setForm(EMPTY);
      flash.success(reply.message);
    } catch (err) {
      const detail = err.payload?.detail;
      if (detail && typeof detail === 'object' && !Array.isArray(detail)) {
        setErrors(detail);
      }
      flash.error(err.message || 'Could not update the password.');
    } finally {
      setBusy(false);
    }
  };

  const errorFor = (field) =>
    errors[field] ? (
      <ul className="field-errors">
        {errors[field].map((message) => (
          <li key={message}>{message}</li>
        ))}
      </ul>
    ) : null;

  if (!user) return null;

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <h1>Your profile</h1>
          <p className="page-sub">Account details and password.</p>
        </div>
        <button type="button" className="btn btn-ghost" onClick={signOut}>
          Sign out
        </button>
      </header>

      <div className="grid-side">
        <div>
          <Card title="Change password">
            <form onSubmit={submit} className="auth-form" noValidate>
              <label className="field">
                Current password
                <input
                  className={`input ${errors.current_password ? 'is-invalid' : ''}`}
                  type="password"
                  value={form.current_password}
                  onChange={update('current_password')}
                  autoComplete="current-password"
                />
                {errorFor('current_password')}
              </label>

              <div className="field-row">
                <label className="field">
                  New password
                  <input
                    className={`input ${errors.new_password ? 'is-invalid' : ''}`}
                    type="password"
                    value={form.new_password}
                    onChange={update('new_password')}
                    autoComplete="new-password"
                  />
                  {errorFor('new_password')}
                </label>

                <label className="field">
                  Confirm new password
                  <input
                    className={`input ${errors.confirm_password ? 'is-invalid' : ''}`}
                    type="password"
                    value={form.confirm_password}
                    onChange={update('confirm_password')}
                    autoComplete="new-password"
                  />
                  {errorFor('confirm_password')}
                </label>
              </div>

              <button
                type="submit"
                className="btn btn-primary"
                disabled={busy || !form.current_password || !form.new_password}
              >
                {busy ? 'Updating…' : 'Update password'}
              </button>

              <p className="muted">
                Changing your password does not sign you out of this browser, but an email is
                sent to {user.email} so a change you did not make cannot pass unnoticed.
              </p>
            </form>
          </Card>
        </div>

        <aside>
          <Card title="Account">
            <dl className="profile-list">
              <dt>Name</dt>
              <dd>{user.full_name}</dd>
              <dt>Username</dt>
              <dd>{user.username}</dd>
              <dt>Email</dt>
              <dd>{user.email}</dd>
              <dt>Status</dt>
              <dd>
                <Badge tone={user.is_active ? 'success' : 'warning'}>
                  {user.is_active ? 'active' : 'not activated'}
                </Badge>
              </dd>
              <dt>Joined</dt>
              <dd>{new Date(user.created_at).toLocaleDateString()}</dd>
              {user.last_login_at && (
                <>
                  <dt>Last sign-in</dt>
                  <dd>{new Date(user.last_login_at).toLocaleString()}</dd>
                </>
              )}
            </dl>
          </Card>

          <Card title="Your corpus">
            <p className="muted">
              {papers.length} paper{papers.length === 1 ? '' : 's'} uploaded,{' '}
              {indexedPapers.length} indexed. Only you can see them.
            </p>
          </Card>
        </aside>
      </div>
    </div>
  );
}
