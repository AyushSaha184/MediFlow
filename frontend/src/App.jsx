import { useState, useEffect, useRef } from "react";
import axios from "axios";
import { AlertTriangle, Play, FolderOpen, Minus, Trash2, CheckCircle, XCircle } from "lucide-react";

// In production (Vercel), VITE_API_URL is set to the Render backend URL.
// In dev, it is empty and Vite's proxy forwards the requests to localhost:8000.
const API_BASE = import.meta.env.VITE_API_URL ?? "";
if (API_BASE) {
  axios.defaults.baseURL = API_BASE;
}

// â”€â”€ Status badge styles â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
const statusStyles = {
  Analyzed:  "text-emerald-400",
  Uploaded:  "text-sky-400",
  Scanning:  "text-amber-300",
  Uploading: "text-violet-300",
  Error:     "text-rose-400",
};

// â”€â”€ Urgency badge colours â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
const urgencyStyles = {
  Low:      "bg-emerald-500/20 text-emerald-300 border-emerald-500/40",
  Medium:   "bg-amber-500/20 text-amber-300 border-amber-500/40",
  High:     "bg-orange-500/20 text-orange-300 border-orange-500/40",
  Critical: "bg-rose-500/20 text-rose-300 border-rose-500/40",
};

function App() {
  const [sessionId, setSessionId]                   = useState(null);
  const sessionIdRef                                 = useRef(null);
  const uploadedFileNamesRef                         = useRef(new Set());
  const [statusRows, setStatusRows]                  = useState([]);
  const [report, setReport]                          = useState(null);
  const [isUploading, setIsUploading]                = useState(false);
  const [isAnalyzing, setIsAnalyzing]                = useState(false);
  const [error, setError]                            = useState(null);
  const [warning, setWarning]                        = useState(null);
  const [showBugHelp, setShowBugHelp]                = useState(false);
  const hasUploadedFiles                             = statusRows.length > 0;
  const analysisCompleted                            = Boolean(report);
  const analysisStarted                              = isAnalyzing || analysisCompleted;

  // â”€â”€ Session lifecycle â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  useEffect(() => {
    let cancelled = false;
    const createSessionWithRetry = async (attemptsLeft = 10, delayMs = 1500) => {
      try {
        const res = await axios.post("/session/create");
        if (!cancelled) {
          setSessionId(res.data.session_id);
          sessionIdRef.current = res.data.session_id;
          setError(null);
        }
      } catch {
        if (cancelled) return;
        if (attemptsLeft <= 1) {
          setError("Failed to create session. Is the backend running?");
          return;
        }
        await new Promise((r) => setTimeout(r, delayMs));
        createSessionWithRetry(attemptsLeft - 1, Math.min(delayMs * 1.5, 8000));
      }
    };
    createSessionWithRetry();

    const handleUnload = () => {
      const sid = sessionIdRef.current;
      if (sid) {
        const base = API_BASE || window.location.origin;
        navigator.sendBeacon(`${base}/session/${sid}/cleanup`);
      }
    };
    window.addEventListener("beforeunload", handleUnload);
    return () => {
      cancelled = true;
      window.removeEventListener("beforeunload", handleUnload);
    };
  }, []);

  // â”€â”€ File upload â†’ /intake â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  const handleFileUpload = async (event) => {
    const allFiles = Array.from(event.target.files ?? []);
    event.target.value = "";
    if (!allFiles.length || !sessionId) return;
    if (analysisCompleted) {
      setWarning("Analysis is already completed for this session. Refresh the page to start a new session.");
      return;
    }

    setError(null);
    setWarning(null);

    // Detect files already ingested this session
    const dupes = allFiles.filter((f) => uploadedFileNamesRef.current.has(f.name));
    const files = allFiles.filter((f) => !uploadedFileNamesRef.current.has(f.name));

    if (dupes.length > 0) {
      setWarning(`Already uploaded: ${dupes.map((f) => f.name).join(", ")}`);
    }
    if (!files.length) return;

    // Add rows immediately as "Scanning"
    const newRows = files.map((f) => ({
      fileName: f.name,
      type: f.name.split(".").pop()?.toUpperCase() ?? "UNKNOWN",
      status: "Scanning",
    }));
    setStatusRows((prev) => [...newRows, ...prev]);
    setIsUploading(true);

    try {
      const form = new FormData();
      files.forEach((f) => form.append("files", f));

      await axios.post(`/intake?session_id=${sessionId}`, form, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      // Track successfully uploaded file names for duplicate detection
      files.forEach((f) => uploadedFileNamesRef.current.add(f.name));

      // Mark uploaded files as Uploaded (privacy scan ran inside intake)
      setStatusRows((prev) =>
        prev.map((row) =>
          newRows.some((n) => n.fileName === row.fileName && row.status === "Scanning")
            ? { ...row, status: "Uploaded" }
            : row
        )
      );
    } catch (err) {
      const msg = err.response?.data?.detail ?? err.message ?? "Upload failed.";
      setError(msg);
      setStatusRows((prev) =>
        prev.map((row) =>
          newRows.some((n) => n.fileName === row.fileName && row.status === "Scanning")
            ? { ...row, status: "Error" }
            : row
        )
      );
    } finally {
      setIsUploading(false);
    }
  };

  // Remove a single error row from UI (file never reached DB)
  const handleDeleteRow = (fileName) => {
    setStatusRows((prev) => prev.filter((r) => r.fileName !== fileName));
  };

  // Clear all rows + diagnosis: wipe DB/bucket for this session, start fresh session
  const handleClearAll = async () => {
    if (analysisStarted) return;
    const sid = sessionIdRef.current;
    setStatusRows([]);
    setReport(null);
    setWarning(null);
    setError(null);
    uploadedFileNamesRef.current = new Set();
    // Wipe pgvector rows, disk files, and Supabase bucket for the old session
    if (sid) {
      try { await axios.delete(`/session/${sid}`); } catch (_) { /* best-effort */ }
    }
    // Create a fresh session so the user can re-upload the same files
    try {
      const res = await axios.post("/session/create");
      setSessionId(res.data.session_id);
      sessionIdRef.current = res.data.session_id;
    } catch (_) {
      setError("Failed to create a new session. Please refresh the page.");
    }
  };

  // â”€â”€ Start analysis â†’ /analyze-medical-session â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  const handleStartAnalysis = async () => {
    if (!sessionId || isUploading) return;
    if (analysisCompleted) {
      setWarning("Analysis is already completed for this session. Refresh the page to run a new analysis.");
      return;
    }
    setError(null);
    setIsAnalyzing(true);

    try {
      const res = await axios.post(`/analyze-medical-session/${sessionId}`);
      setReport(res.data);
      setStatusRows((prev) =>
        prev.map((row) =>
          row.status === "Uploaded" ? { ...row, status: "Analyzed" } : row
        )
      );
    } catch (err) {
      const msg = err.response?.data?.detail ?? err.message ?? "Analysis failed.";
      setError(msg);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // â”€â”€ Loading splash â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  if (!sessionId && !error) {
    return (
      <div className="flex h-screen flex-col items-center justify-center gap-3 bg-[#0f172a] text-slate-400 text-sm tracking-widest">
        <div className="h-5 w-5 animate-spin rounded-full border-2 border-amber-400 border-t-transparent" />
        <span>Initialising session...</span>
        <span className="text-xs text-slate-600">(first load may take up to 60 s while the backend wakes up)</span>
      </div>
    );
  }

  const diag = report?.structured_diagnosis;

  return (
    <div className="h-screen overflow-hidden bg-[#0f172a] text-slate-100">
      {/* Background glows */}
      <div className="pointer-events-none fixed -top-24 left-1/4 h-72 w-72 rounded-full bg-fuchsia-500/10 blur-3xl" />
      <div className="pointer-events-none fixed top-20 right-10 h-80 w-80 rounded-full bg-violet-500/10 blur-3xl" />

      {/* Header */}
      <header className="fixed inset-x-0 top-0 z-20 h-14 border-b border-slate-700/60 bg-slate-900/95 px-5">
        <div className="flex h-full items-center justify-between">
          <span className="text-sm font-semibold tracking-[0.22em] text-amber-300">MEDIFLOW AI</span>
          <div
            className="relative"
            onMouseEnter={() => setShowBugHelp(true)}
            onMouseLeave={() => setShowBugHelp(false)}
          >
            <a
              href="mailto:ayushsaha1884@gmail.com?subject=MediFlow%20Bug%20Report"
              className="inline-flex items-center gap-1.5 rounded-full border border-slate-500/60 bg-slate-800/80 px-3 py-1 text-sm font-medium text-slate-100 transition hover:border-amber-300 hover:text-amber-200"
            >
              <AlertTriangle size={14} />
              Report Bug
            </a>
            {showBugHelp && (
              <div className="absolute right-0 mt-2 w-64 rounded-2xl border border-slate-600/60 bg-slate-900/95 px-4 py-3 text-sm text-slate-200 shadow-xl">
                Please report the bug{" "}
                <a
                  href="mailto:ayushsaha1884@gmail.com?subject=MediFlow%20Bug%20Report"
                  className="text-amber-300 underline"
                >
                  here
                </a>{" "}
                with screenshots.
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Error banner */}
      {error && (
        <div className="fixed inset-x-0 top-14 z-30 flex items-center gap-3 bg-rose-900/80 px-5 py-2 text-sm text-rose-200 backdrop-blur-sm">
          <AlertTriangle size={15} className="shrink-0" />
          <span>{error}</span>
          <button onClick={() => setError(null)} className="ml-auto text-rose-300 hover:text-white">✕</button>
        </div>
      )}

      <main className={`flex h-full min-h-0 pb-8 ${error ? "pt-24" : "pt-14"}`}>
        {/* â”€â”€ Left: Upload panel â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
        <section className="w-1/2 border-r border-slate-700/50 p-4">
          <div className="h-full rounded-xl border border-slate-700/60 bg-slate-800/40 p-4">
            <div className="mb-3 flex items-center justify-between gap-2">
              <h2 className="text-lg font-semibold tracking-wide text-slate-200">Upload</h2>
              <div className="flex items-center gap-2">
                {hasUploadedFiles && !analysisStarted && (
                  <button
                    type="button"
                    onClick={handleClearAll}
                    disabled={isUploading || isAnalyzing}
                    className="inline-flex items-center gap-1.5 rounded-md border border-slate-600 px-3 py-1.5 text-xs font-semibold text-slate-300 transition hover:border-rose-400 hover:text-rose-300 disabled:opacity-50"
                    title="Delete all uploaded files from the database and start fresh"
                  >
                    <Trash2 size={13} />
                    Clear
                  </button>
                )}
                <button
                  type="button"
                  onClick={handleStartAnalysis}
                  disabled={analysisStarted || isUploading || !hasUploadedFiles}
                  className="inline-flex items-center gap-2 rounded-md bg-amber-400 px-3 py-1.5 text-xs font-semibold text-slate-900 transition hover:bg-amber-300 disabled:opacity-50"
                >
                  <Play size={14} />
                  {isAnalyzing ? "Analyzing..." : analysisCompleted ? "Analysis Complete" : "Start Analysis"}
                </button>
              </div>
            </div>

            {/* Drop zone */}
            <label
              onClick={(e) => {
                if (analysisCompleted) {
                  e.preventDefault();
                  setWarning("This session is locked after analysis. Refresh the page to upload again.");
                }
              }}
              className={`mb-4 flex h-48 flex-col items-center justify-center rounded-xl border border-slate-500/80 bg-slate-900/30 text-center transition ${
                analysisCompleted
                  ? "cursor-not-allowed opacity-60"
                  : "cursor-pointer hover:border-amber-300 hover:bg-slate-900/50"
              }`}
            >
              <FolderOpen size={72} className="mb-2 text-slate-200" />
              <span className="text-xl font-medium tracking-wide text-amber-300">
                {isUploading ? "Uploadingâ€¦" : "Upload Documents"}
              </span>
              <span className="mt-1 text-xs text-slate-500">PDF · DICOM · JPEG · PNG · ZIP</span>
              <input
                type="file"
                className="hidden"
                multiple
                disabled={isUploading || analysisCompleted}
                accept=".zip,.dcm,.dicom,.pdf,.jpg,.jpeg,.png"
                onChange={handleFileUpload}
              />
            </label>

            {/* Duplicate warning */}
            {warning && (
              <div className="mb-3 flex items-center gap-2 rounded-md border border-amber-500/40 bg-amber-900/20 px-3 py-2 text-xs text-amber-300">
                <AlertTriangle size={13} className="shrink-0" />
                <span>{warning}</span>
                <button onClick={() => setWarning(null)} className="ml-auto hover:text-amber-100">✕</button>
              </div>
            )}

            {/* File table */}
            <div className="rounded-lg border border-slate-700/60 bg-slate-900/35">
              <div className="grid grid-cols-[1.8fr_0.6fr_0.9fr_1.5rem] border-b border-slate-700/70 px-3 py-2 text-xs font-semibold uppercase tracking-widest text-slate-400">
                <span>File Name</span>
                <span>Type</span>
                <span>Status</span>
                <span />
              </div>
              <div className="max-h-[calc(100vh-26rem)] overflow-y-auto">
                {statusRows.length === 0 && (
                  <p className="px-3 py-4 text-center text-xs text-slate-600">No files uploaded yet.</p>
                )}
                {statusRows.map((row, idx) => (
                  <div
                    key={`${row.fileName}-${idx}`}
                    className="grid grid-cols-[1.8fr_0.6fr_0.9fr_1.5rem] items-center border-b border-slate-700/40 px-3 py-2 text-sm text-slate-200 last:border-b-0"
                  >
                    <span className="truncate pr-2">{row.fileName}</span>
                    <span className="text-rose-300">{row.type}</span>
                    <span className={`inline-flex items-center gap-1.5 ${statusStyles[row.status] ?? "text-slate-300"}`}>
                      {(row.status === "Scrubbed" || row.status === "Analyzed") && (
                        <CheckCircle size={13} className="text-amber-400" />
                      )}
                      {row.status === "Error" && <XCircle size={13} />}
                      {row.status}
                    </span>
                    <span>
                      {row.status === "Error" && (
                        <button
                          onClick={() => handleDeleteRow(row.fileName)}
                          className="text-slate-500 hover:text-rose-400 transition"
                          title="Remove from list"
                        >
                          <Trash2 size={13} />
                        </button>
                      )}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* â”€â”€ Right: Diagnosis panel â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
        <section className="w-1/2 min-h-0 overflow-y-auto p-4">
          <div className="space-y-4 rounded-xl border border-violet-400/20 bg-slate-900/40 p-4 backdrop-blur-sm">
            <h2 className="text-lg font-semibold tracking-wide text-slate-200">Diagnosis</h2>

            {/* Empty state — how to use */}
            {!hasUploadedFiles && !report && (
              <div className="flex min-h-[calc(100vh-15rem)] items-center justify-center">
                <div className="w-full max-w-md space-y-3">
                  <p className="mb-4 text-center text-sm font-semibold tracking-widest text-fuchsia-300 uppercase">How to use MediFlow</p>
                  {[
                    { step: "1", text: "Upload one or more patient files using the panel on the left. Accepted formats: PDF, DICOM, JPEG, PNG, or ZIP archives." },
                    { step: "2", text: "Wait for files to be scanned and privacy-scrubbed. Status will update to \"Scrubbed\" when ready." },
                    { step: "3", text: 'Click \"Start Analysis\" to run the full multi-agent AI pipeline — RAG retrieval, diagnostic reasoning, and explainability.' },
                    { step: "4", text: "The AI-generated diagnosis, differential diagnoses, clinician brief, and patient explanation will appear here." },
                  ].map(({ step, text }) => (
                    <div key={step} className="flex items-start gap-3 rounded-xl border border-slate-700/50 bg-slate-800/40 px-4 py-3">
                      <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-amber-400 text-[10px] font-bold text-slate-900">{step}</span>
                      <p className="text-sm leading-relaxed text-slate-300">{text}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Waiting for analysis */}
            {hasUploadedFiles && !report && !isAnalyzing && (
              <p className="py-8 text-center text-sm text-slate-500">
                Files staged. Click <span className="text-amber-300">Start Analysis</span> to run the AI pipeline.
              </p>
            )}

            {/* Analyzing spinner */}
            {isAnalyzing && (
              <div className="flex flex-col items-center gap-3 py-12">
                <div className="h-8 w-8 animate-spin rounded-full border-2 border-amber-400 border-t-transparent" />
                <p className="text-sm text-slate-400 tracking-wide">Running multi-agent pipeline...</p>
              </div>
            )}

            {/* â”€â”€ Full report â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
            {report && diag && (
              <>
                {/* Primary diagnosis + urgency */}
                <article className="rounded-lg border border-slate-700/60 bg-slate-900/40 p-4">
                  <p className="mb-2 text-xs font-semibold uppercase tracking-widest text-slate-400">Primary Diagnosis</p>
                  <div className="flex flex-wrap items-start justify-between gap-2">
                    <p className="text-xl font-semibold text-slate-100">{diag.primary_diagnosis}</p>
                    <span className={`rounded-full border px-3 py-0.5 text-xs font-semibold ${urgencyStyles[diag.urgency_level] ?? urgencyStyles.Medium}`}>
                      {diag.urgency_level}
                    </span>
                  </div>
                  <p className="mt-2 text-xs text-slate-500">
                    Confidence: <span className="text-slate-300">{(diag.confidence_score * 100).toFixed(0)}%</span>
                  </p>
                </article>

                {/* Differentials */}
                {diag.differential_diagnoses?.length > 0 && (
                  <article className="rounded-lg border border-slate-700/60 bg-slate-900/40 p-4">
                    <p className="mb-2 text-xs font-semibold uppercase tracking-widest text-slate-400">Differential Diagnoses</p>
                    <ul className="space-y-1">
                      {diag.differential_diagnoses.map((d, i) => (
                        <li key={i} className="flex items-start gap-2 text-sm text-slate-200">
                          <Minus size={14} className="mt-0.5 shrink-0 text-violet-400" />{d}
                        </li>
                      ))}
                    </ul>
                  </article>
                )}

                {/* Contraindications */}
                {diag.contraindications?.length > 0 && (
                  <article className="rounded-lg border border-rose-700/40 bg-rose-950/20 p-4">
                    <p className="mb-2 text-xs font-semibold uppercase tracking-widest text-rose-400">âš  Contraindications</p>
                    <ul className="space-y-1">
                      {diag.contraindications.map((c, i) => (
                        <li key={i} className="text-sm text-rose-200">{c}</li>
                      ))}
                    </ul>
                  </article>
                )}

                {/* Clinician brief */}
                <article className="rounded-lg border border-slate-700/60 bg-slate-900/40 p-4">
                  <p className="mb-2 text-xs font-semibold uppercase tracking-widest text-slate-400">Clinician Brief</p>
                  <p className="text-sm leading-relaxed text-slate-100">{report.clinician_brief}</p>
                </article>

                {/* Patient explanation */}
                <article className="rounded-lg border border-slate-700/60 bg-slate-900/40 p-4">
                  <p className="mb-2 text-xs font-semibold uppercase tracking-widest text-slate-400">Patient-Friendly Explanation</p>
                  <p className="text-sm leading-relaxed text-slate-200">{report.patient_explanation}</p>
                </article>

                {/* Missing data */}
                {diag.missing_data_points?.length > 0 && (
                  <article className="rounded-lg border border-amber-700/30 bg-amber-950/20 p-4">
                    <p className="mb-2 text-xs font-semibold uppercase tracking-widest text-amber-400">Recommended Additional Tests</p>
                    <ul className="space-y-1">
                      {diag.missing_data_points.map((m, i) => (
                        <li key={i} className="text-sm text-amber-200">{m}</li>
                      ))}
                    </ul>
                  </article>
                )}

                {/* Evidence table */}
                {report.evidence_table?.length > 0 && (
                  <article className="rounded-lg border border-slate-700/60 bg-slate-900/40 p-4">
                    <p className="mb-3 text-xs font-semibold uppercase tracking-widest text-slate-400">Evidence Sources</p>
                    <ul className="space-y-3 text-sm text-slate-200">
                      {report.evidence_table.map((item, i) => (
                        <li
                          key={i}
                          className={`rounded-md border p-3 ${item.is_contradictory ? "border-rose-700/50 bg-rose-950/20" : "border-slate-700/60 bg-slate-950/40"}`}
                        >
                          {item.is_contradictory && (
                            <span className="mb-1 inline-block rounded bg-rose-800/50 px-1.5 py-0.5 text-[10px] font-semibold uppercase text-rose-300">
                              Contradictory
                            </span>
                          )}
                          <p className="mb-1">{item.statement}</p>
                          <p className="text-xs text-slate-500">
                            {item.source_type} · confidence {(item.confidence_of_mapping * 100).toFixed(0)}%
                            {item.source_chunk_ids?.length > 0 && ` · chunks: ${item.source_chunk_ids.join(", ")}`}
                          </p>
                        </li>
                      ))}
                    </ul>
                  </article>
                )}

                {/* Citations */}
                {report.citations?.length > 0 && (
                  <article className="rounded-lg border border-slate-700/60 bg-slate-900/40 p-4">
                    <p className="mb-2 text-xs font-semibold uppercase tracking-widest text-slate-400">Citations</p>
                    <ol className="list-decimal list-inside space-y-1">
                      {report.citations.map((c, i) => (
                        <li key={i} className="text-xs text-slate-400">{c}</li>
                      ))}
                    </ol>
                  </article>
                )}

                {/* Disclaimer */}
                <p className="rounded-md bg-slate-800/60 px-3 py-2 text-[11px] leading-relaxed text-slate-500">
                  {report.disclaimer}
                </p>
              </>
            )}
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="fixed inset-x-0 bottom-0 z-20 h-8 border-t border-slate-700/60 bg-slate-900/95 px-4">
        <div className="flex h-full items-center justify-center text-[11px] tracking-wide text-slate-300">
          Disclaimer: AI can make mistakes, please consult a doctor first.
        </div>
      </footer>
    </div>
  );
}

export default App;
