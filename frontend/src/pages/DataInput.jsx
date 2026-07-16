import { useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { MdUploadFile, MdCheckCircle, MdInsertDriveFile } from 'react-icons/md'
import Header from '../components/Header'
import { fadeIn } from '../constants/animations'
import { uploadDataset } from '../services/api'

// Columns info for the reference card
const REQUIRED_COLS = [
  'Moisture_Content_%',
  'Fat_Content_%',
  'Protein_Content_%',
  'Water_Activity_Aw',
  'Acid_Insoluble_Ash_%',
  'Acid_Value_mgKOHg',
  'Peroxide_Value',
  'Total_Plate_Count_CFUg',
  'Yeast_Mold_Count_CFUg',
  'Lead_Pb_mgkg',
  'Cadmium_Cd_mgkg',
  'Product_Name',
]

const OPTIONAL_COLS = ['Sample_ID', 'Batch_Code', 'Test_Date']

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
}

export default function DataInput() {
  const navigate = useNavigate()
  const fileInputRef = useRef(null)

  const [selectedFile, setSelectedFile] = useState(null)
  const [isDragging, setIsDragging] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [resultMessage, setResultMessage] = useState(null) // { message, samples }
  const [errorMessage, setErrorMessage] = useState(null)

  // ── File selection helpers ────────────────────────────────────────────────
  const validateAndSet = (file) => {
    setResultMessage(null)
    setErrorMessage(null)

    if (!file) return

    if (!file.name.toLowerCase().endsWith('.csv')) {
      setErrorMessage('Invalid file type. Please select a .csv file.')
      return
    }

    setSelectedFile(file)
  }

  const handleFileChange = (e) => {
    validateAndSet(e.target.files?.[0] ?? null)
  }

  // ── Drag-and-drop handlers ────────────────────────────────────────────────
  const handleDragOver = (e) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    setIsDragging(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    validateAndSet(e.dataTransfer.files?.[0] ?? null)
  }

  // ── Upload ────────────────────────────────────────────────────────────────
  const handleUpload = async () => {
    if (!selectedFile || isUploading) return

    setIsUploading(true)
    setErrorMessage(null)
    setResultMessage(null)

    try {
      const response = await uploadDataset(selectedFile)
      const { message, samples } = response.data
      setResultMessage({ message, samples })
      setSelectedFile(null)
      if (fileInputRef.current) fileInputRef.current.value = ''
    } catch (err) {
      const msg =
        err?.response?.data?.detail ||
        err?.response?.data?.message ||
        err?.message ||
        'Upload failed. Please try again.'
      setErrorMessage(msg)
    } finally {
      setIsUploading(false)
    }
  }

  return (
    <motion.div {...fadeIn}>
      <Header
        breadcrumbs={['Layer 3', 'Data Input']}
        title="Data Input"
        subtitle="Upload a CSV dataset to re-process predictions, SHAP values, and cluster assignments across the entire pipeline."
      />

      <div className="content-inner">

        {/* ── Upload Card ──────────────────────────────────────────────────── */}
        <div className="clean-card section-gap">
          <div className="sec-title">Upload Dataset</div>

          {/* Drag-and-drop zone */}
          <div
            className={`upload-dropzone${isDragging ? ' dragging' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => !selectedFile && fileInputRef.current?.click()}
            role="button"
            tabIndex={0}
            aria-label="Upload CSV file drop zone"
            onKeyDown={(e) => {
              if ((e.key === 'Enter' || e.key === ' ') && !selectedFile) {
                fileInputRef.current?.click()
              }
            }}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              style={{ display: 'none' }}
              onChange={handleFileChange}
              id="csv-file-input"
            />

            {!selectedFile ? (
              <>
                <MdUploadFile
                  style={{
                    fontSize: 48,
                    color: isDragging ? 'var(--c-navy)' : '#BDBDCA',
                    marginBottom: 12,
                    transition: 'color 0.2s',
                  }}
                />
                <div
                  style={{
                    fontSize: '0.9rem',
                    fontWeight: 600,
                    color: isDragging ? 'var(--c-navy)' : '#555',
                    marginBottom: 6,
                    transition: 'color 0.2s',
                  }}
                >
                  {isDragging ? 'Drop your CSV file here' : 'Drag & drop your CSV file here'}
                </div>
                <div style={{ fontSize: '0.75rem', color: '#888', marginBottom: 14 }}>
                  or
                </div>
                <button
                  type="button"
                  className="btn-primary"
                  style={{ margin: '0 auto', height: 36, padding: '0 22px' }}
                  onClick={(e) => {
                    e.stopPropagation()
                    fileInputRef.current?.click()
                  }}
                >
                  Browse File
                </button>
              </>
            ) : (
              /* File selected state */
              <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
                <MdInsertDriveFile style={{ fontSize: 36, color: 'var(--c-navy)', flexShrink: 0 }} />
                <div style={{ textAlign: 'left' }}>
                  <div
                    style={{ fontWeight: 700, fontSize: '0.9rem', color: 'var(--c-navy)', marginBottom: 2 }}
                  >
                    {selectedFile.name}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: '#888' }}>
                    {formatBytes(selectedFile.size)}
                  </div>
                </div>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation()
                    setSelectedFile(null)
                    setErrorMessage(null)
                    setResultMessage(null)
                    if (fileInputRef.current) fileInputRef.current.value = ''
                  }}
                  style={{
                    marginLeft: 'auto',
                    background: 'none',
                    border: '1px solid var(--c-border)',
                    borderRadius: 2,
                    padding: '4px 10px',
                    fontSize: '0.75rem',
                    color: '#777',
                    cursor: 'pointer',
                  }}
                >
                  Remove
                </button>
              </div>
            )}
          </div>

          {/* Error message */}
          {errorMessage && (
            <div
              role="alert"
              style={{
                marginTop: 14,
                padding: '10px 14px',
                background: '#FDEDEC',
                border: '1px solid #F5B7B1',
                borderRadius: 2,
                fontSize: '0.8rem',
                color: '#E74C3C',
                fontWeight: 500,
              }}
            >
              {errorMessage}
            </div>
          )}

          {/* Success message */}
          {resultMessage && (
            <div
              role="status"
              style={{
                marginTop: 14,
                padding: '14px 16px',
                background: '#EAF8F0',
                border: '1px solid #A9DFBF',
                borderRadius: 2,
                display: 'flex',
                flexDirection: 'column',
                gap: 10,
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <MdCheckCircle style={{ fontSize: 20, color: '#2ECC71', flexShrink: 0 }} />
                <div style={{ fontWeight: 700, fontSize: '0.85rem', color: '#27AE60' }}>
                  Upload Successful
                </div>
              </div>
              <div style={{ fontSize: '0.8rem', color: '#27AE60' }}>
                {resultMessage.message
                  ? resultMessage.message
                  : `Dataset processed — ${resultMessage.samples ?? '?'} samples loaded.`}
                {resultMessage.samples && resultMessage.message && (
                  <span style={{ marginLeft: 6, fontWeight: 700 }}>
                    ({resultMessage.samples} samples)
                  </span>
                )}
              </div>
              <button
                type="button"
                className="btn-primary"
                style={{ width: 'fit-content', height: 34, padding: '0 18px', marginTop: 2 }}
                onClick={() => navigate('/')}
              >
                Go to Dashboard
              </button>
            </div>
          )}

          {/* Upload button */}
          {selectedFile && !resultMessage && (
            <div style={{ marginTop: 14 }}>
              <button
                type="button"
                id="upload-process-btn"
                className="btn-primary"
                disabled={isUploading}
                onClick={handleUpload}
                style={{ height: 38, padding: '0 28px', fontSize: '0.85rem', marginLeft: 0 }}
              >
                {isUploading ? 'Processing…' : 'Upload & Process'}
              </button>
            </div>
          )}
        </div>

        {/* ── Column Format Reference Card ─────────────────────────────────── */}
        <div className="clean-card">
          <div className="sec-title">Supported Column Format</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>

            {/* Required columns */}
            <div>
              <div
                style={{
                  fontSize: '0.7rem',
                  fontWeight: 700,
                  textTransform: 'uppercase',
                  letterSpacing: '0.6px',
                  color: 'var(--c-navy)',
                  marginBottom: 10,
                }}
              >
                Required Columns
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {REQUIRED_COLS.map((col) => (
                  <div
                    key={col}
                    style={{
                      background: '#F4F5F8',
                      border: '1px solid var(--c-border)',
                      borderRadius: 2,
                      padding: '5px 10px',
                      fontSize: '0.75rem',
                      fontFamily: 'monospace',
                      color: '#333',
                    }}
                  >
                    {col}
                  </div>
                ))}
              </div>
            </div>

            {/* Optional columns + notes */}
            <div>
              <div
                style={{
                  fontSize: '0.7rem',
                  fontWeight: 700,
                  textTransform: 'uppercase',
                  letterSpacing: '0.6px',
                  color: '#888',
                  marginBottom: 10,
                }}
              >
                Optional Columns
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginBottom: 20 }}>
                {OPTIONAL_COLS.map((col) => (
                  <div
                    key={col}
                    style={{
                      background: '#FAFAFA',
                      border: '1px dashed var(--c-border)',
                      borderRadius: 2,
                      padding: '5px 10px',
                      fontSize: '0.75rem',
                      fontFamily: 'monospace',
                      color: '#888',
                    }}
                  >
                    {col}
                  </div>
                ))}
              </div>

              {/* Notes */}
              <div
                style={{
                  background: '#FFFBEB',
                  border: '1px solid #FDE68A',
                  borderRadius: 2,
                  padding: '10px 14px',
                  fontSize: '0.75rem',
                  color: '#92400E',
                  lineHeight: 1.7,
                }}
              >
                <div style={{ fontWeight: 700, marginBottom: 4 }}>ℹ Notes</div>
                <ul style={{ paddingLeft: 16, margin: 0 }}>
                  <li>If <code>Sample_ID</code>, <code>Batch_Code</code>, or <code>Test_Date</code> are empty, they will be auto-generated by the backend.</li>
                  <li>Only <strong>.csv</strong> files are accepted.</li>
                  <li>Column names are case-sensitive. Use exact names as shown.</li>
                  <li>Numeric columns must not contain non-numeric characters (e.g., units).</li>
                </ul>
              </div>
            </div>

          </div>
        </div>

      </div>
    </motion.div>
  )
}
