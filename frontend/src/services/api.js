/**
 * BQIS API Service
 * ----------------
 * All chart and page data flows through this module.
 * Currently returns mock JSON. When the Python backend is connected,
 * swap `useMock` to false and ensure Flask/FastAPI endpoints are running.
 *
 * Placeholder endpoints:
 *   GET /api/dashboard
 *   GET /api/risk-overview
 *   GET /api/shap
 *   GET /api/clusters
 *   GET /api/executive-summary
 */

import axios from 'axios'
import dashboardData  from '../mock/dashboard.json'
import riskData       from '../mock/risk.json'
import shapData       from '../mock/shap.json'
import clustersData   from '../mock/clusters.json'
import summaryData    from '../mock/summary.json'

const USE_MOCK = false // Set false when Python backend is live

const api = axios.create({
  baseURL: '/api',
  timeout: 10000,
  headers: { 'Content-Type': 'application/json' },
})

/** Simulate async network call for mock data */
const mockDelay = (data, ms = 80) =>
  new Promise(resolve => setTimeout(() => resolve({ data }), ms))

// ─── Filter Options ──────────────────────────────────────────────────────────
export const fetchFilterOptions = () =>
  USE_MOCK
    ? mockDelay({
        batches: ["BCH-07-003", "BCH-07-001", "BCH-06-012"],
        products: ["Butter Biscuit", "Marie Biscuit"]
      })
    : api.get('/filters/options')

// ─── Dashboard ───────────────────────────────────────────────────────────────
export const fetchDashboard = (params) =>
  USE_MOCK ? mockDelay(dashboardData) : api.get('/dashboard', { params })

// ─── Sample Risk Overview ─────────────────────────────────────────────────────
export const fetchRiskOverview = (params) =>
  USE_MOCK ? mockDelay(riskData) : api.get('/risk-overview', { params })

// ─── SHAP Parameter Importance ────────────────────────────────────────────────
export const fetchShap = () =>
  USE_MOCK ? mockDelay(shapData) : api.get('/shap')

// ─── Failure Clusters / PCA ───────────────────────────────────────────────────
export const fetchClusters = (params) =>
  USE_MOCK ? mockDelay(clustersData) : api.get('/clusters', { params })

// ─── Executive Summary ────────────────────────────────────────────────────────
export const fetchExecutiveSummary = (params) =>
  USE_MOCK ? mockDelay(summaryData) : api.get('/executive-summary', { params })

// ─── Download Report ──────────────────────────────────────────────────────────
export const downloadReport = (type, params) =>
  api.get(`/report/${type}`, { params, responseType: 'blob' })
