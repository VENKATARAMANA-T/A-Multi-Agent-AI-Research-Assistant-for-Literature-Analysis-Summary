import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import api, { getAccessToken, setAccessToken, setUnauthorizedHandler } from '../api/client';

const AuthContext = createContext(null);

/**
 * Who is signed in, and the three operations that change it.
 *
 * The stored token is not trusted on its own: `ready` stays false until
 * `/api/auth/me` has confirmed it. Rendering the app on the strength of a
 * string in localStorage would flash the whole interface before an expired
 * token bounced the user back to sign-in.
 */
export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [ready, setReady] = useState(false);

  const signOut = useCallback(() => {
    setAccessToken(null);
    setUser(null);
  }, []);

  // A 401 from any request ends the session once, centrally, rather than each
  // page discovering the expiry on its own.
  useEffect(() => {
    setUnauthorizedHandler(() => {
      setAccessToken(null);
      setUser(null);
    });
    return () => setUnauthorizedHandler(null);
  }, []);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      if (!getAccessToken()) {
        if (!cancelled) setReady(true);
        return;
      }
      try {
        const profile = await api.me();
        if (!cancelled) setUser(profile);
      } catch {
        setAccessToken(null);
        if (!cancelled) setUser(null);
      } finally {
        if (!cancelled) setReady(true);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  const adopt = useCallback((payload) => {
    setAccessToken(payload.access_token);
    setUser(payload.user);
    return payload.user;
  }, []);

  const signIn = useCallback(
    async (identifier, password) => adopt(await api.login({ identifier, password })),
    [adopt],
  );

  const value = useMemo(
    () => ({
      user,
      ready,
      isAuthenticated: Boolean(user),
      signIn,
      signOut,
      register: (payload) => api.register(payload),
      refresh: async () => setUser(await api.me()),
    }),
    [user, ready, signIn, signOut],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) throw new Error('useAuth must be used inside an AuthProvider');
  return context;
}
