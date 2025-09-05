import React, { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import LoaderComponent from "./LoaderComponent"

const rootElement = document.getElementById("root")

if (!rootElement) {
  throw new Error("Root element not found")
}

const root = createRoot(rootElement)

root.render(
  <StrictMode>
    <LoaderComponent />
  </StrictMode>
)
