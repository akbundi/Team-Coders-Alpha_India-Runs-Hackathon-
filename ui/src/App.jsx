import { useState, useRef } from 'react'
import Papa from 'papaparse'
import { BrainCircuit, Loader2, UploadCloud, ShieldAlert, FileSpreadsheet, LayoutGrid, List } from 'lucide-react'
import './App.css'

function App() {
  const [candidates, setCandidates] = useState([])
  const [loading, setLoading] = useState(false)
  const [analyzed, setAnalyzed] = useState(false)
  const [viewMode, setViewMode] = useState('leaderboard') // 'leaderboard' or 'grid'
  const fileInputRef = useRef(null)

  // Handle Drag and Drop
  const handleDragOver = (e) => {
    e.preventDefault()
  }

  const handleDrop = (e) => {
    e.preventDefault()
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processFile(e.dataTransfer.files[0])
    }
  }

  // Handle File Input Selection
  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      processFile(e.target.files[0])
    }
  }

  // Process the CSV File
  const processFile = (file) => {
    setLoading(true)
    setTimeout(() => { // Fake delay for UX
      Papa.parse(file, {
        header: true,
        skipEmptyLines: true,
        complete: (results) => {
          setCandidates(results.data)
          setLoading(false)
          setAnalyzed(true)
        },
        error: (err) => {
          console.error("Failed to parse CSV", err)
          setLoading(false)
        }
      })
    }, 1500)
  }

  const getScoreColor = (scoreStr) => {
    const score = parseFloat(scoreStr)
    if (score >= 0.8) return 'high'
    if (score >= 0.5) return 'medium'
    return 'low'
  }

  const getScoreGradient = (scoreStr) => {
    const score = parseFloat(scoreStr)
    if (score >= 0.8) return 'linear-gradient(90deg, #10B981, #34D399)'
    if (score >= 0.5) return 'linear-gradient(90deg, #F59E0B, #FBBF24)'
    return 'linear-gradient(90deg, #EF4444, #F87171)'
  }

  return (
    <div className="app-container">
      <header className="header">
        <div className="header-title">
          <BrainCircuit size={40} color="#06B6D4" />
          <h1>Redrob AI Recruiter</h1>
        </div>
        
        {analyzed && (
          <div className="header-stats">
            <div className="stat-box">
              <div className="stat-value">{candidates.length.toLocaleString()}</div>
              <div className="stat-label">Candidates</div>
            </div>
            <button 
              className="upload-btn" 
              onClick={() => {
                setAnalyzed(false)
                setCandidates([])
              }}
            >
              Upload New Data
            </button>
          </div>
        )}
      </header>

      <main>
        {/* Upload State */}
        {!analyzed && !loading && (
          <div 
            className="upload-zone"
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current.click()}
          >
            <input 
              type="file" 
              accept=".csv" 
              style={{display: 'none'}} 
              ref={fileInputRef}
              onChange={handleFileSelect}
            />
            <UploadCloud size={64} color="#3B82F6" />
            <h2>Upload Dataset</h2>
            <p>Drag and drop your `submission.csv` or click to browse files.</p>
            <button className="upload-btn">
              <FileSpreadsheet size={18} style={{marginRight: '8px', verticalAlign: 'middle'}}/>
              Select CSV File
            </button>
          </div>
        )}

        {/* Loading State */}
        {loading && (
          <div className="loading-container">
            <Loader2 className="spinner" size={64} />
            <h2>Processing Signals...</h2>
            <p>Evaluating TF-IDF vectors, Redrob behavioral signals, and Honeypot metrics.</p>
          </div>
        )}

        {/* Analyzed State */}
        {analyzed && !loading && (
          <>
            <div className="tabs">
              <button 
                className={`tab-btn ${viewMode === 'leaderboard' ? 'active' : ''}`}
                onClick={() => setViewMode('leaderboard')}
              >
                <List size={16} style={{marginRight: '6px', verticalAlign: 'text-bottom'}} />
                Leaderboard
              </button>
              <button 
                className={`tab-btn ${viewMode === 'grid' ? 'active' : ''}`}
                onClick={() => setViewMode('grid')}
              >
                <LayoutGrid size={16} style={{marginRight: '6px', verticalAlign: 'text-bottom'}} />
                Detailed Analysis
              </button>
            </div>

            {/* Leaderboard View */}
            {viewMode === 'leaderboard' && (
              <table className="leaderboard-table">
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>Candidate ID</th>
                    <th style={{width: '30%'}}>Composite Score</th>
                    <th>Signals</th>
                  </tr>
                </thead>
                <tbody>
                  {candidates.map((cand, idx) => {
                    const rank = cand.rank || (idx + 1)
                    const scoreFloat = parseFloat(cand.score)
                    const isHoneypot = scoreFloat === 0.0 || cand.reasoning?.includes('FRAUD') || cand.reasoning?.includes('framework') || cand.reasoning?.includes('stuffing')
                    
                    return (
                      <tr key={cand.candidate_id || idx} className="leaderboard-row">
                        <td>
                          <div className={`rank-circle ${rank <= 3 ? `rank-${rank}` : ''}`}>
                            {rank}
                          </div>
                        </td>
                        <td style={{fontWeight: 600}}>
                          {cand.candidate_id}
                        </td>
                        <td>
                          <div style={{fontWeight: 800, color: isHoneypot ? '#EF4444' : '#fff'}}>
                            {(scoreFloat * 100).toFixed(1)}%
                          </div>
                          <div className="score-bar-container">
                            <div 
                              className="score-bar" 
                              style={{
                                width: `${scoreFloat * 100}%`,
                                background: getScoreGradient(cand.score)
                              }}
                            />
                          </div>
                        </td>
                        <td>
                          <div className="badges">
                            {isHoneypot && <span className="badge danger">Fraud Detected</span>}
                            {!isHoneypot && cand.reasoning?.includes('startup') && <span className="badge success">Startup Fit</span>}
                            {!isHoneypot && cand.reasoning?.includes('No major concerns') && <span className="badge success">Clean Record</span>}
                            {!isHoneypot && cand.reasoning?.includes('responsive') && <span className="badge">Responsive</span>}
                          </div>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            )}

            {/* Grid View */}
            {viewMode === 'grid' && (
              <div className="grid-container">
                {candidates.map((cand, idx) => {
                  const isHoneypot = parseFloat(cand.score) === 0.0 || cand.reasoning?.includes('FRAUD') || cand.reasoning?.includes('framework') || cand.reasoning?.includes('stuffing')
                  
                  return (
                    <div key={cand.candidate_id || idx} className={`candidate-card ${isHoneypot ? 'honeypot' : ''}`}>
                      <div className="card-header">
                        <div className="card-title">
                          <span className="rank-badge">Rank #{cand.rank || (idx + 1)}</span>
                          <h3>{cand.candidate_id}</h3>
                        </div>
                        <div className="score-container">
                          <div className={`score-value ${getScoreColor(cand.score)}`}>
                            {(parseFloat(cand.score) * 100).toFixed(1)}%
                          </div>
                          <div className="score-label">Composite Score</div>
                        </div>
                      </div>
                      
                      <div className={`reasoning-box ${isHoneypot ? 'fraud-box' : ''}`}>
                        {isHoneypot && <ShieldAlert size={16} style={{marginBottom: '0.5rem'}}/>}
                        <p>{cand.reasoning}</p>
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </>
        )}
      </main>
    </div>
  )
}

export default App
