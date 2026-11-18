/** Entry point of the offline demo build: same app, mocked backend, in-memory routing. */
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { MemoryRouter } from "react-router-dom";
import App from "../App";
import "../index.css";
import { installMockFetch } from "./mockFetch";

installMockFetch();

// MemoryRouter: the demo may run inside a sandboxed frame where URL routing is
// unavailable. A "#/route" in the URL still selects the start page.
const initialPath = window.location.hash.replace(/^#/, "") || "/testing";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <MemoryRouter initialEntries={[initialPath]}>
      <App />
    </MemoryRouter>
    <div className="fixed bottom-3 right-3 z-50 max-w-xs rounded-md bg-amber-100 text-amber-900 text-xs px-3 py-2 border border-amber-300 shadow">
      Mode démo : données fictives, backend simulé dans le navigateur. Aucun appel LLM ni clé API.
    </div>
  </StrictMode>,
);
