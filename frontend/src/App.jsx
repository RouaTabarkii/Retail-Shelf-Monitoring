import { useState, useRef, useEffect } from 'react'
import './App.css'

const API_URL = "http://127.0.0.1:8000/predict"
const INTERVAL_MS = 1000   // une analyse par seconde (vidéo et webcam)
const IOU_MIN = 0.3        // recouvrement minimal pour dire "c'est le même gap"
const RAYON_NAME = "Rayon Épicerie - Pâtes & Huiles"   // nom fictif

// Recouvrement entre deux rectangles (0 = aucun, 1 = identiques)
function overlap(a, b) {
  const ix = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1))
  const iy = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1))
  const inter = ix * iy
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1)
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1)
  const union = areaA + areaB - inter
  return union > 0 ? inter / union : 0
}

function App() {
  const [mode, setMode] = useState("image")          // "image" | "video" | "webcam"
  const [file, setFile] = useState(null)
  const [mediaUrl, setMediaUrl] = useState(null)     // adresse de l'image ou de la vidéo choisie
  const [stream, setStream] = useState(null)         // flux de la webcam
  const [lastResult, setLastResult] = useState(null)
  const [detectedGaps, setDetectedGaps] = useState([])
  const [latencyMs, setLatencyMs] = useState(null)
  const [analysedCount, setAnalysedCount] = useState(0)
  const [skippedCount, setSkippedCount] = useState(0)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const videoRef = useRef(null)
  const timerRef = useRef(null)
  const busyRef = useRef(false)
  const previousGapsRef = useRef([])
  const idRef = useRef(0)
  const streamRef = useRef(null)

  // ---------- Remises à zéro ----------
  function resetAll() {
    stopAnalysis()
    previousGapsRef.current = []
    setFile(null)
    setMediaUrl(null)
    setLastResult(null)
    setDetectedGaps([])
    setLatencyMs(null)
    setAnalysedCount(0)
    setSkippedCount(0)
    setLoading(false)
    setError(null)
  }

  function resetMemory() {
    previousGapsRef.current = []
    setLastResult(null)
  }

  function switchMode(newMode) {
    if (newMode === mode) return
    stopWebcam()
    resetAll()
    setMode(newMode)
  }

  // ---------- Choix d'un fichier (image ou vidéo) ----------
  function handleChange(e) {
    const selected = e.target.files[0]
    if (!selected) return

    resetAll()
    setFile(selected)
    setMediaUrl(URL.createObjectURL(selected))
  }

  // ---------- Webcam ----------
  async function startWebcam() {
    resetAll()
    try {
      const s = await navigator.mediaDevices.getUserMedia({ video: true, audio: false })
      streamRef.current = s
      setStream(s)
    } catch (err) {
      console.error(err)
      setError("Impossible d'accéder à la caméra. Vérifie l'autorisation du navigateur.")
    }
  }

  function stopWebcam() {
    stopAnalysis()
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop())
      streamRef.current = null
    }
    setStream(null)
  }

  // Branche le flux sur la balise <video> dès qu'elle existe
  useEffect(() => {
    if (stream && videoRef.current) {
      videoRef.current.srcObject = stream
      videoRef.current.play().catch(() => {})
    }
  }, [stream])

  // ---------- Capture d'une image de la vidéo ----------
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

  // Ajoute une ligne au tableau seulement pour les gaps qui viennent d'apparaître
  function registerNewGaps(data, instant) {
    const previous = previousGapsRef.current
    const current = []
    const newRows = []

    for (const gap of data.gaps) {
      const rect = {
        x1: gap.gap_start,
        x2: gap.gap_end,
        y1: gap.y_haut,
        y2: gap.y_bas,
      }

      const alreadySeen = previous.some((p) => overlap(p, rect) >= IOU_MIN)
      current.push(rect)

      if (!alreadySeen) {
        idRef.current += 1
        newRows.push({
          id: idRef.current,
          instant,
          etagere: gap.etagere,
          ecart_px: gap.ecart_px,
        })
      }
    }

    previousGapsRef.current = current

    if (newRows.length > 0) {
      setDetectedGaps((prev) => [...prev, ...newRows])
    }
  }

  // ---------- Envoi d'une image à FastAPI (commun aux trois modes) ----------
  async function sendBlob(blob, instant) {
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

    registerNewGaps(data, instant)
    setLastResult({ ...data, instant })
    setAnalysedCount((n) => n + 1)
    setError(null)
  }

  // Mode image : une seule analyse, au clic
  async function analyzeImage() {
    if (!file) return

    setLoading(true)
    setError(null)
    setDetectedGaps([])
    previousGapsRef.current = []

    try {
      await sendBlob(file, null)
    } catch (err) {
      console.error(err)
      setError("Impossible de joindre l'API. Vérifie que le serveur est lancé.")
    } finally {
      setLoading(false)
    }
  }

  // Modes vidéo et webcam : une analyse à chaque tick du minuteur
  async function analyzeFrame() {
    if (busyRef.current) {
      setSkippedCount((n) => n + 1)
      return
    }
    busyRef.current = true

    try {
      const video = videoRef.current
      if (!video) return
      const instant = video.currentTime
      const blob = await captureFrame()
      await sendBlob(blob, instant)
    } catch (err) {
      console.error(err)
      setError("Impossible de joindre l'API. Vérifie que le serveur est lancé.")
    } finally {
      busyRef.current = false
    }
  }

  function startAnalysis() {
    if (timerRef.current) return
    analyzeFrame()
    timerRef.current = setInterval(analyzeFrame, INTERVAL_MS)
  }

  function stopAnalysis() {
    clearInterval(timerRef.current)
    timerRef.current = null
  }

  // Au démontage : arrête le minuteur et coupe la caméra
  useEffect(() => {
    return () => {
      clearInterval(timerRef.current)
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop())
      }
    }
  }, [])

  const showMedia = mode === "webcam" ? stream : mediaUrl

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
          <h2>Source</h2>

          <div className="tabs">
            <button
              className={`tab ${mode === "image" ? "tab-active" : ""}`}
              onClick={() => switchMode("image")}
            >
              Image
            </button>
            <button
              className={`tab ${mode === "video" ? "tab-active" : ""}`}
              onClick={() => switchMode("video")}
            >
              Vidéo
            </button>
            <button
              className={`tab ${mode === "webcam" ? "tab-active" : ""}`}
              onClick={() => switchMode("webcam")}
            >
              Webcam
            </button>
          </div>

          <div className="upload-row">
            {mode !== "webcam" && (
              <>
                <label className="btn btn-secondary">
                  <input
                    key={mode}
                    type="file"
                    accept={mode === "image" ? "image/*" : "video/*"}
                    onChange={handleChange}
                    hidden
                  />
                  {mode === "image" ? "Choisir une image" : "Choisir une vidéo"}
                </label>
                <span className="file-name">
                  {file ? file.name : "Aucun fichier choisi"}
                </span>
              </>
            )}

            {mode === "image" && (
              <button
                className="btn btn-primary btn-analyze"
                disabled={!file || loading}
                onClick={analyzeImage}
              >
                {loading ? "Analyse en cours..." : "Analyser"}
              </button>
            )}

            {mode === "webcam" && (
              <>
                {!stream ? (
                  <button className="btn btn-primary" onClick={startWebcam}>
                    Démarrer la caméra
                  </button>
                ) : (
                  <button className="btn btn-secondary" onClick={stopWebcam}>
                    Arrêter la caméra
                  </button>
                )}
                <span className="file-name">
                  {stream ? "Caméra active" : "Caméra éteinte"}
                </span>
              </>
            )}
          </div>

          {error && <p className="error">{error}</p>}
        </section>

        {showMedia && (
          <section className="card">
            <div className="legend">
              <span><i className="dot dot-green" /> Produit</span>
              <span><i className="dot dot-red" /> Gap</span>
              <span><i className="dot dot-blue" /> Personne</span>
            </div>

            <div className="media-box">
              {mode === "image" ? (
                <img src={mediaUrl} alt="Image du rayon" />
              ) : (
                <video
                  ref={videoRef}
                  src={mode === "video" ? mediaUrl : undefined}
                  controls={mode === "video"}
                  muted={mode === "webcam"}
                  playsInline
                  onPlay={startAnalysis}
                  onPause={stopAnalysis}
                  onEnded={stopAnalysis}
                  onSeeking={resetMemory}
                />
              )}

              {lastResult && (
                <svg
                  viewBox={`0 0 ${lastResult.image_width} ${lastResult.image_height}`}
                >
                  {Object.values(lastResult.etageres).flat().map((box, index) => (
                    <rect
                      key={`p-${index}`}
                      x={box[0]}
                      y={box[1]}
                      width={box[2] - box[0]}
                      height={box[3] - box[1]}
                      fill="none"
                      stroke="lime"
                      strokeWidth="3"
                    />
                  ))}

                  {lastResult.gaps.map((gap, index) => (
                    <rect
                      key={`g-${index}`}
                      x={gap.gap_start}
                      y={gap.y_haut}
                      width={gap.gap_end - gap.gap_start}
                      height={gap.y_bas - gap.y_haut}
                      fill="red"
                      fillOpacity="0.3"
                      stroke="red"
                      strokeWidth="4"
                    />
                  ))}

                  {(lastResult.personnes || []).map((b, i) => (
                    <rect
                      key={`h-${i}`}
                      x={b[0]}
                      y={b[1]}
                      width={b[2] - b[0]}
                      height={b[3] - b[1]}
                      fill="none"
                      stroke="dodgerblue"
                      strokeWidth="4"
                      strokeDasharray="10 6"
                    />
                  ))}
                </svg>
              )}
            </div>
          </section>
        )}

        {lastResult && (
          <section className="card">
            <h2>Dernière analyse</h2>
            <div className="stats">
              <div className="stat">
                <span className="stat-value stat-text">{RAYON_NAME}</span>
                <span className="stat-label">Nom du rayon</span>
              </div>
              <div className="stat">
                <span className="stat-value">{lastResult.num_boxes}</span>
                <span className="stat-label">Produits détectés</span>
              </div>
              <div className={`stat ${lastResult.gaps.length > 0 ? "stat-alert" : "stat-ok"}`}>
                <span className="stat-value">{lastResult.gaps.length}</span>
                <span className="stat-label">Produits absents</span>
              </div>
              <div className="stat">
                <span className="stat-value">{latencyMs} ms</span>
                <span className="stat-label">Temps de réponse</span>
              </div>
            </div>
            <p style={{ color: "#6b7280", fontSize: 14, margin: 0 }}>
              {analysedCount} analyses faites
              {mode !== "image" && ` · ${skippedCount} images sautées`}
            </p>
          </section>
        )}

        {showMedia && (
          <section className="card">
            <h2>Absences détectées</h2>
            {detectedGaps.length === 0 ? (
              <p style={{ color: "#6b7280", margin: 0 }}>
                Aucune absence détectée pour le moment.
              </p>
            ) : (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Détecté à (s)</th>
                      <th>Étagère</th>
                      <th>Écart (px)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {detectedGaps.map((g) => (
                      <tr key={g.id}>
                        <td>{g.id}</td>
                        <td>{g.instant == null ? "-" : g.instant.toFixed(1)}</td>
                        <td>{g.etagere + 1}</td>
                        <td>{g.ecart_px}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        )}
      </main>

    </div>
  )
}

export default App
