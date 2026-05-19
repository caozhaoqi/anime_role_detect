import axios from 'axios'

const api = axios.create({
  baseURL: '/api/auth',
  timeout: 10000
})

// 添加请求拦截器，自动添加 token
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

export const authApi = {
  login: (data) => api.post('/login', data),
  
  register: (data) => api.post('/register', data),
  
  logout: () => api.post('/logout'),
  
  getProfile: () => api.get('/me')
}

export default api
