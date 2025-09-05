import React, { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import FiberComponent from "./FiberComponent"

const rootElement = document.getElementById("root")

if (!rootElement) {
  throw new Error("Root element not found")
}

const root = createRoot(rootElement)

root.render(
  <StrictMode>
    <FiberComponent />
  </StrictMode>
)
