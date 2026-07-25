import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';

interface ApiClientOptions {
  baseURL?: string;
  timeout?: number;
  onUnauthorized?: () => void;
  onLoadingChange?: (loading: boolean) => void;
}

class ApiClient {
  private instance: AxiosInstance;
  private requestCount = 0;
  private abortControllers = new Map<string, AbortController>();
  private onUnauthorized?: () => void;
  private onLoadingChange?: (loading: boolean) => void;

  constructor(options: ApiClientOptions = {}) {
    this.onUnauthorized = options.onUnauthorized;
    this.onLoadingChange = options.onLoadingChange;

    this.instance = axios.create({
      baseURL: options.baseURL || '/api',
      timeout: options.timeout || 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors(): void {
    this.instance.interceptors.request.use(
      (config) => {
        this.incrementRequestCount();

        const token = localStorage.getItem('accessToken');
        if (token && config.headers) {
          config.headers.Authorization = `Bearer ${token}`;
        }

        if (config.url) {
          const controller = new AbortController();
          config.signal = controller.signal;
          this.abortControllers.set(config.url, controller);
        }

        return config;
      },
      (error) => {
        this.decrementRequestCount();
        return Promise.reject(error);
      }
    );

    this.instance.interceptors.response.use(
      (response) => {
        this.decrementRequestCount();
        return response;
      },
      async (error) => {
        this.decrementRequestCount();

        if (error.response?.status === 401) {
          await this.handleUnauthorized(error);
        }

        return Promise.reject(error);
      }
    );
  }

  private incrementRequestCount(): void {
    this.requestCount++;
    if (this.requestCount === 1) {
      this.onLoadingChange?.(true);
    }
  }

  private decrementRequestCount(): void {
    this.requestCount--;
    if (this.requestCount === 0) {
      this.onLoadingChange?.(false);
    }
  }

  private async handleUnauthorized(error: any): Promise<void> {
    const originalRequest = error.config;

    if (originalRequest._retry) {
      this.onUnauthorized?.();
      return;
    }

    originalRequest._retry = true;

    const refreshToken = localStorage.getItem('refreshToken');
    if (!refreshToken) {
      this.onUnauthorized?.();
      return;
    }

    try {
      const formData = new FormData();
      formData.append('refresh_token', refreshToken);

      const response = await axios.post('/api/auth/refresh', formData);

      if (response.data.success) {
        const { access_token, refresh_token } = response.data.data;

        localStorage.setItem('accessToken', access_token);
        localStorage.setItem('refreshToken', refresh_token);

        originalRequest.headers.Authorization = `Bearer ${access_token}`;
        return this.instance(originalRequest);
      }
    } catch {
      this.onUnauthorized?.();
    }

    this.onUnauthorized?.();
  }

  abortAllRequests(): void {
    this.abortControllers.forEach((controller) => controller.abort());
    this.abortControllers.clear();
  }

  abortRequest(url: string): void {
    const controller = this.abortControllers.get(url);
    if (controller) {
      controller.abort();
      this.abortControllers.delete(url);
    }
  }

  get<T = any>(url: string, config?: AxiosRequestConfig): Promise<AxiosResponse<T>> {
    return this.instance.get<T>(url, config);
  }

  post<T = any>(url: string, data?: any, config?: AxiosRequestConfig): Promise<AxiosResponse<T>> {
    return this.instance.post<T>(url, data, config);
  }

  put<T = any>(url: string, data?: any, config?: AxiosRequestConfig): Promise<AxiosResponse<T>> {
    return this.instance.put<T>(url, data, config);
  }

  delete<T = any>(url: string, config?: AxiosRequestConfig): Promise<AxiosResponse<T>> {
    return this.instance.delete<T>(url, config);
  }

  getInstance(): AxiosInstance {
    return this.instance;
  }
}

export const createApiClient = (options?: ApiClientOptions): ApiClient => {
  return new ApiClient(options);
};

export let apiClient: ApiClient;

export const initApiClient = (options?: ApiClientOptions): void => {
  apiClient = createApiClient(options);
};