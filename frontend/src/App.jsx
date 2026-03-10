import { useState } from "react";
import axios from "axios";
import { ArrowDown, FolderOpen, Play } from "lucide-react";

const statusStyles = {
  Analyzed: "text-emerald-400",
  Scrubbed: "text-sky-400",
  Scanning: "text-amber-300"
};

function App() {
  const [sessionId] = useState("default-session");
  const [statusRows, setStatusRows] = useState([]);
  const [finalDiagnosticReport, setFinalDiagnosticReport] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const hasUploadedFiles = statusRows.length > 0;

  const handleFileUpload = async (event) => {
    const files = Array.from(event.target.files ?? []);
    if (!files.length || !sessionId.trim()) {
      return;
    }

    const newRows = files.map((file) => {
      const extension = file.name.split(".").pop()?.toUpperCase() ?? "UNKNOWN";
      return { fileName: file.name, type: extension, status: "Scanning" };
    });
    setStatusRows((prev) => [...newRows, ...prev]);

    // Stub upload orchestration call requested by backend contract.
    try {
      await axios.post(`/analyze-medical-session/${sessionId}`);
    } catch (error) {
      console.error("handleFileUpload stub failed:", error);
    } finally {
      event.target.value = "";
    }
  };

  const handleStartAnalysis = async () => {
    if (!sessionId.trim()) {
      return;
    }

    setIsAnalyzing(true);
    try {
      const response = await axios.post(`/analyze-medical-session/${sessionId}`);
      if (response?.data) {
        setFinalDiagnosticReport(response.data);
      }
      setStatusRows((prev) =>
        prev.map((row) =>
          row.status === "Scanning" ? { ...row, status: "Analyzed" } : row
        )
      );
    } catch (error) {
      console.error("Macro-chain orchestration failed:", error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="h-screen overflow-hidden bg-[#0f172a] text-slate-100">
      <div className="pointer-events-none fixed -top-24 left-1/4 h-72 w-72 rounded-full bg-fuchsia-500/10 blur-3xl" />
      <div className="pointer-events-none fixed top-20 right-10 h-80 w-80 rounded-full bg-violet-500/10 blur-3xl" />
      <header className="fixed inset-x-0 top-0 z-20 h-14 border-b border-slate-700/60 bg-slate-900/95 px-5">
        <div className="flex h-full items-center">
          <span className="text-sm font-semibold tracking-[0.22em] text-amber-300">
            MEDIFLOW AI
          </span>
        </div>
      </header>

      <main className="flex h-full min-h-0 pt-14 pb-8">
        <section className="w-1/2 border-r border-slate-700/50 p-4">
          <div className="h-full rounded-xl border border-slate-700/60 bg-slate-800/40 p-4">
            <div className="mb-3 flex items-center justify-between gap-2">
              <h2 className="text-lg font-semibold tracking-wide text-slate-200">
                Upload
              </h2>
              <button
                type="button"
                onClick={handleStartAnalysis}
                disabled={isAnalyzing || !hasUploadedFiles}
                className="inline-flex items-center gap-2 rounded-md bg-amber-400 px-3 py-1.5 text-xs font-semibold text-slate-900 transition hover:bg-amber-300 disabled:opacity-50"
              >
                <Play size={14} />
                {isAnalyzing ? "Analyzing..." : "Start Analysis"}
              </button>
            </div>

            <label className="mb-4 flex h-56 cursor-pointer flex-col items-center justify-center rounded-xl border border-dashed border-slate-500/80 bg-slate-900/30 text-center transition hover:border-amber-300 hover:bg-slate-900/50">
              <FolderOpen size={86} className="mb-2 text-slate-200" />
              <span className="text-2xl font-medium tracking-wide text-amber-300">
                Upload Documents
              </span>
              <input
                type="file"
                className="hidden"
                multiple
                accept=".zip,.dcm,.dicom,.pdf,.jpg,.jpeg,.png"
                onChange={handleFileUpload}
              />
            </label>

            <div className="rounded-lg border border-slate-700/60 bg-slate-900/35">
              <div className="grid grid-cols-[1.8fr_0.7fr_0.7fr] border-b border-slate-700/70 px-3 py-2 text-xs font-semibold uppercase tracking-widest text-slate-400">
                <span>File Name</span>
                <span>Type</span>
                <span>Status</span>
              </div>
              <div className="max-h-[calc(100vh-26rem)] overflow-y-auto">
                {statusRows.map((row, idx) => (
                  <div
                    key={`${row.fileName}-${idx}`}
                    className="grid grid-cols-[1.8fr_0.7fr_0.7fr] border-b border-slate-700/40 px-3 py-2 text-sm text-slate-200 last:border-b-0"
                  >
                    <span className="truncate">{row.fileName}</span>
                    <span className="text-rose-300">{row.type}</span>
                    <span className={statusStyles[row.status] || "text-slate-300"}>
                      {row.status}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        <section className="w-1/2 min-h-0 overflow-y-auto p-4">
          <div className="space-y-4 rounded-xl border border-violet-400/20 bg-slate-900/40 p-4 backdrop-blur-sm">
            <h2 className="text-lg font-semibold tracking-wide text-slate-200">
              Diagnosis
            </h2>

            {!hasUploadedFiles ? (
              <div className="flex min-h-[calc(100vh-15rem)] items-center justify-center">
                <div className="grid w-full max-w-2xl grid-cols-1 gap-3 md:grid-cols-3">
                  <article className="rounded-2xl border border-violet-300/25 bg-white/10 p-4 text-center backdrop-blur-sm">
                    <p className="text-sm font-semibold tracking-wide text-fuchsia-300">
                      MediFlow Assistant
                    </p>
                  </article>
                  <article className="rounded-2xl border border-violet-300/25 bg-white/10 p-4 text-center backdrop-blur-sm">
                    <p className="text-sm font-semibold tracking-wide text-slate-100">
                      How can I assist you today?
                    </p>
                  </article>
                  <article className="rounded-2xl border border-violet-300/25 bg-white/10 p-4 text-center backdrop-blur-sm">
                    <p className="text-sm font-semibold tracking-wide text-slate-100">
                      Upload Documents
                    </p>
                  </article>
                </div>
              </div>
            ) : (
              <>
                <article className="rounded-lg border border-slate-700/60 bg-slate-900/40 p-4">
                  <p className="mb-2 text-xs font-semibold uppercase tracking-widest text-slate-400">
                    Labs
                  </p>
                  <div className="flex items-center justify-between">
                    <span className="text-2xl font-medium text-slate-100">Hemoglobin</span>
                    <span className="flex items-center gap-2 text-2xl font-semibold text-rose-400">
                      9.2
                      <ArrowDown size={22} />
                    </span>
                  </div>
                </article>

                <article className="rounded-lg border border-slate-700/60 bg-slate-900/40 p-4">
                  <p className="mb-2 text-xs font-semibold uppercase tracking-widest text-slate-400">
                    Diagnostic Summary
                  </p>
                  <p className="mb-4 text-sm leading-relaxed text-slate-100">
                    {finalDiagnosticReport?.clinician_brief}
                  </p>
                  <p className="mb-2 text-xs font-semibold uppercase tracking-widest text-slate-400">
                    Patient-Friendly Explanation
                  </p>
                  <p className="text-sm leading-relaxed text-slate-200">
                    {finalDiagnosticReport?.patient_explanation}
                  </p>
                </article>

                <article className="rounded-lg border border-slate-700/60 bg-slate-900/40 p-4">
                  <p className="mb-3 text-xs font-semibold uppercase tracking-widest text-slate-400">
                    Sources
                  </p>
                  <ul className="space-y-3 text-sm text-slate-200">
                    {finalDiagnosticReport?.evidence_table?.map((item, index) => (
                      <li
                        key={`${item.statement}-${index}`}
                        className="rounded-md border border-slate-700/60 bg-slate-950/40 p-3"
                      >
                        <p className="mb-1">{item.statement}</p>
                        <p className="text-xs text-slate-400">
                          source_chunk_ids: {(item.source_chunk_ids || []).join(", ")}
                        </p>
                      </li>
                    ))}
                  </ul>
                </article>
              </>
            )}
          </div>
        </section>
      </main>

      <footer className="fixed inset-x-0 bottom-0 z-20 h-8 border-t border-slate-700/60 bg-slate-900/95 px-4">
        <div className="flex h-full items-center justify-center text-[11px] tracking-wide text-slate-300">
          Disclaimer: AI can make mistakes, please consult a doctor first.
        </div>
      </footer>
    </div>
  );
}

export default App;
