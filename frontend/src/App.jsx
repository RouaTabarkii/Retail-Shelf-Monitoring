import { useState, useRef, useEffect } from 'react'
import './App.css'

const API_URL = "http://127.0.0.1:8000/predict"
const INTERVAL_MS = 1000   // une analyse par seconde

function App() {
  const [file, setFile] = useState(null)
  const [videoUrl, setVideoUrl] = useState(null)
  const [lastResult, setLastResult] = useState(null)
  const [latencyMs, setLatencyMs] = useState(null)
  const [analysedCount, setAnalysedCount] = useState(0)
  const [skippedCount, setSkippedCount] = useState(0)
  const [error, setError] = useState(null)

  const videoRef = useRef(null)
  const timerRef = useRef(null)
  const busyRef = useRef(false)

  function handleChange(e) {
    const selected = e.target.files[0]
    if (!selected) return

    stopAnalysis()
    setFile(selected)
    setVideoUrl(URL.createObjectURL(selected))
    setLastResult(null)
    setLatencyMs(null)
    setAnalysedCount(0)
    setSkippedCount(0)
    setError(null)
  }

  // Capture l'image affichée par la vidéo, sous forme de fichier (Blob)
  function captureFrame() {
    return new Promise((resolve, reject) => {
      const video = videoRef.current
      if (!video || video.videoWidth === 0) {
        reject(new Error("La vidéo n'est pas encore chargée"))
        return
      }

      const canvas = document.createElement("canvas")
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height)

      canvas.toBlob(
        (blob) => (blob ? resolve(blob) : reject(new Error("Capture impossible"))),
        "image/jpeg",
        0.9
      )
    })
  }

  // Une analyse : capture, envoi, réponse
  async function analyzeFrame() {
    if (busyRef.current) {
      setSkippedCount((n) => n + 1)   // l'analyse précédente n'est pas finie
      return
    }
    busyRef.current = true

    try {
      const instant = videoRef.current.currentTime
      const blob = await captureFrame()

      const formData = new FormData()
      formData.append("file", blob, "frame.jpg")

      const start = performance.now()
      const response = await fetch(API_URL, { method: "POST", body: formData })

      if (!response.ok) {
        console.error("Erreur API :", response.status)
        setError(`Erreur du serveur (code ${response.status})`)
        return
      }

      const data = await response.json()
      setLatencyMs(Math.round(performance.now() - start))
      setLastResult({ ...data, instant })
      setAnalysedCount((n) => n + 1)
      setError(null)
    } catch (err) {
      console.error(err)
      setError("Impossible de joindre l'API. Vérifie que le serveur est lancé.")
    } finally {
      busyRef.current = false
    }
  }

  function startAnalysis() {
    if (timerRef.current) return          // déjà démarré
    analyzeFrame()                        // une analyse immédiate
    timerRef.current = setInterval(analyzeFrame, INTERVAL_MS)
  }

  function stopAnalysis() {
    clearInterval(timerRef.current)
    timerRef.current = null
  }

  // Arrête le minuteur si le composant disparaît
  useEffect(() => stopAnalysis, [])

  return (
    <div className="page">
      <header className="topbar">
        <div className="topbar-inner">
          <div>
            <h1>Retail Shelf Monitoring</h1>
            <span className="subtitle">Surveillance des rayons</span>
          </div>
          <span className="store">Monoprix</span>
        </div>
      </header>

      <main className="content">
        <section className="card">
          <h2>Analyser une vidéo</h2>
          <div className="upload-row">
            <label className="btn btn-secondary">
              <input type="file" accept="video/*" onChange={handleChange} hidden />
              Choisir une vidéo
            </label>
            <span className="file-name">
              {file ? file.name : "Aucun fichier choisi"}
            </span>
          </div>
          {error && <p className="error">{error}</p>}
        </section>

        {videoUrl && (
          <section className="card">
            <video
              ref={videoRef}
              src={videoUrl}
              controls
              onPlay={startAnalysis}
              onPause={stopAnalysis}
              onEnded={stopAnalysis}
              style={{ display: "block", width: "100%", borderRadius: 8 }}
            />
          </section>
        )}

        {lastResult && (
          <section className="card">
            <h2>Dernière analyse</h2>
            <div className="stats">
              <div className="stat">
                <span className="stat-value">{lastResult.instant.toFixed(1)} s</span>
                <span className="stat-label">Instant de la vidéo</span>
              </div>
              <div className="stat">
                <span className="stat-value">{lastResult.num_boxes}</span>
                <span className="stat-label">Produits détectés</span>
              </div>
              <div className={`stat ${lastResult.gaps.length > 0 ? "stat-alert" : "stat-ok"}`}>
                <span className="stat-value">{lastResult.gaps.length}</span>
                <span className="stat-label">Gaps détectés</span>
              </div>
              <div className="stat">
                <span className="stat-value">{latencyMs} ms</span>
                <span className="stat-label">Temps de réponse</span>
              </div>
            </div>
            <p style={{ color: "#6b7280", fontSize: 14, margin: 0 }}>
              {analysedCount} analyses faites · {skippedCount} images sautées
            </p>
          </section>
        )}
      </main>
    </div>
  )
}

export default App