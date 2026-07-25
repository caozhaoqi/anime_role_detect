import { useState, useCallback, useEffect } from 'react';
import { AuthState } from '../types';
import { AuthService } from '../api/services/AuthService';

export const useAuth = () => {
  const [authState, setAuthState] = useState<AuthState>({
    isAuthenticated: false,
    user: null,
    accessToken: null,
    refreshToken: null,
  });
  const [loginError, setLoginError] = useState<string | null>(null);
  const [isLoginLoading, setIsLoginLoading] = useState(false);
  const [showSessionExpired, setShowSessionExpired] = useState(false);

  const handleUnauthorized = useCallback(() => {
    AuthService.logout();
    setAuthState({
      isAuthenticated: false,
      user: null,
      accessToken: null,
      refreshToken: null,
    });
    setShowSessionExpired(true);
    setTimeout(() => setShowSessionExpired(false), 5000);
  }, []);

  const validateAndSetAuth = useCallback(async () => {
    const savedAccessToken = localStorage.getItem('accessToken');
    const savedRefreshToken = localStorage.getItem('refreshToken');
    const savedUser = localStorage.getItem('currentUser');

    if (savedAccessToken && savedRefreshToken && savedUser) {
      try {
        const isValid = await AuthService.validateToken(savedAccessToken);
        if (isValid) {
          setAuthState({
            isAuthenticated: true,
            user: JSON.parse(savedUser),
            accessToken: savedAccessToken,
            refreshToken: savedRefreshToken,
          });
        } else {
          AuthService.logout();
        }
      } catch {
        AuthService.logout();
      }
    }
  }, []);

  useEffect(() => {
    validateAndSetAuth();
  }, [validateAndSetAuth]);

  const handleLogin = useCallback(async (username: string, password: string) => {
    setIsLoginLoading(true);
    setLoginError(null);

    try {
      const response = await AuthService.login(username, password);

      if (response.success && response.data) {
        const { access_token, refresh_token, username: userName, role } = response.data;
        const user = { username: userName, role };

        setAuthState({
          isAuthenticated: true,
          user,
          accessToken: access_token,
          refreshToken: refresh_token,
        });

        localStorage.setItem('accessToken', access_token);
        localStorage.setItem('refreshToken', refresh_token);
        localStorage.setItem('currentUser', JSON.stringify(user));
      } else {
        setLoginError(response.message || '登录失败');
      }
    } catch {
      setLoginError('登录失败，请稍后重试');
    } finally {
      setIsLoginLoading(false);
    }
  }, []);

  const handleRegister = useCallback(async (username: string, password: string) => {
    setIsLoginLoading(true);
    setLoginError(null);

    try {
      const response = await AuthService.register(username, password);

      if (response.success && response.data) {
        const { access_token, refresh_token, username: userName, role } = response.data;
        const user = { username: userName, role };

        setAuthState({
          isAuthenticated: true,
          user,
          accessToken: access_token,
          refreshToken: refresh_token,
        });

        localStorage.setItem('accessToken', access_token);
        localStorage.setItem('refreshToken', refresh_token);
        localStorage.setItem('currentUser', JSON.stringify(user));
      } else {
        setLoginError(response.message || '注册失败');
      }
    } catch {
      setLoginError('注册失败，请稍后重试');
    } finally {
      setIsLoginLoading(false);
    }
  }, []);

  const handleLogout = useCallback(() => {
    AuthService.logout();
    setAuthState({
      isAuthenticated: false,
      user: null,
      accessToken: null,
      refreshToken: null,
    });
  }, []);

  return {
    authState,
    loginError,
    isLoginLoading,
    showSessionExpired,
    handleLogin,
    handleRegister,
    handleLogout,
    handleUnauthorized,
  };
};