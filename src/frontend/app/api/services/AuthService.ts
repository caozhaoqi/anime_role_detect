import { apiClient } from '../client';
import { LoginRequest, LoginResponse } from '../../types';

export class AuthService {
  static async login(username: string, password: string): Promise<LoginResponse> {
    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);

    const response = await apiClient.post<LoginResponse>('/auth/login', formData);
    return response.data;
  }

  static async register(username: string, password: string): Promise<LoginResponse> {
    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);

    const response = await apiClient.post<LoginResponse>('/auth/register', formData);
    return response.data;
  }

  static async refreshToken(refreshToken: string): Promise<LoginResponse> {
    const formData = new FormData();
    formData.append('refresh_token', refreshToken);

    const response = await apiClient.post<LoginResponse>('/auth/refresh', formData);
    return response.data;
  }

  static async validateToken(token: string): Promise<boolean> {
    try {
      const response = await apiClient.get('/health', {
        headers: { Authorization: `Bearer ${token}` },
      });
      return response.status === 200;
    } catch {
      return false;
    }
  }

  static logout(): void {
    localStorage.removeItem('accessToken');
    localStorage.removeItem('refreshToken');
    localStorage.removeItem('currentUser');
    apiClient.abortAllRequests();
  }
}