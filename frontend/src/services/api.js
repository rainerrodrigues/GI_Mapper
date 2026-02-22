import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:3000/api/v1'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor for adding auth token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => Promise.reject(error)
)

// Response interceptor for handling errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

// Clustering API
export const clusteringAPI = {
  detect: (data) => api.post('/clusters/detect', data),
  getAll: () => api.get('/clusters'),
  getById: (id) => api.get(`/clusters/${id}`),
}

// ROI Prediction API
export const roiAPI = {
  predict: (data) => api.post('/predictions/roi', data),
  getAll: () => api.get('/predictions'),
  getById: (id) => api.get(`/predictions/${id}`),
  getExplanation: (id) => api.get(`/predictions/${id}/explanation`),
}

// Anomaly Detection API
export const anomalyAPI = {
  detect: (data) => api.post('/anomalies/detect', data),
  getAll: () => api.get('/anomalies'),
  getById: (id) => api.get(`/anomalies/${id}`),
  getExplanation: (id) => api.get(`/anomalies/${id}/explanation`),
}

// MLA Scoring API
export const mlaAPI = {
  compute: (data) => api.post('/mla-scores/compute', data),
  getAll: () => api.get('/mla-scores'),
  getByConstituency: (id) => api.get(`/mla-scores/${id}`),
  getExplanation: (id) => api.get(`/mla-scores/${id}/explanation`),
}

// Forecasting API
export const forecastAPI = {
  generate: (data) => api.post('/forecasts/generate', data),
  getAll: () => api.get('/forecasts'),
  getById: (id) => api.get(`/forecasts/${id}`),
}

// Risk Assessment API
export const riskAPI = {
  assess: (data) => api.post('/risk/assess', data),
  getAll: () => api.get('/risk'),
  getById: (id) => api.get(`/risk/${id}`),
}

// Model Performance API
export const performanceAPI = {
  getAll: () => api.get('/models/performance'),
  getByModel: (modelId) => api.get(`/models/${modelId}/metrics`),
}

// GI Products API
export const giProductsAPI = {
  getAll: () => api.get('/gi-products'),
  getById: (id) => api.get(`/gi-products/${id}`),
  create: (data) => api.post('/gi-products', data),
  update: (id, data) => api.put(`/gi-products/${id}`, data),
  delete: (id) => api.delete(`/gi-products/${id}`),
  getByRegion: (bounds) => api.get(`/gi-products/region/${bounds}`),
}

export default api
