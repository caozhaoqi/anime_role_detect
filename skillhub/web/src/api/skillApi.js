import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 10000
})

export const skillApi = {
  getSkills: (params = {}) => api.get('/skills', { params }),
  
  getSkill: (skillId, params = {}) => api.get(`/skills/${skillId}`, { params }),
  
  createSkill: (data) => api.post('/skills', data),
  
  deleteSkill: (skillId) => api.delete(`/skills/${skillId}`),
  
  installSkill: (skillId, params = {}) => api.post(`/skills/${skillId}/install`, null, { params }),
  
  uninstallSkill: (skillId) => api.delete(`/skills/${skillId}/uninstall`),
  
  getVersions: (skillId) => api.get(`/skills/${skillId}/versions`),
  
  searchSkills: (params = {}) => api.get('/search', { params }),
  
  getTags: () => api.get('/tags'),
  
  getCategories: () => api.get('/categories'),
  
  getStats: () => api.get('/stats')
}

export default api
