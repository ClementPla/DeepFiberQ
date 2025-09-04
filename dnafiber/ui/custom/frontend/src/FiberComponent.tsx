import {
  Streamlit,
  withStreamlitConnection,
  ComponentProps,
} from "streamlit-component-lib"

import React, {
  useCallback,
  useEffect,
  useMemo,
  useState,
  ReactElement,
  useRef,
} from "react"

import "./style.css"

import { Provider as StyletronProvider } from "styletron-react"
import { Client as Styletron } from "styletron-engine-atomic"
import { BaseProvider, LightTheme, DarkTheme } from "baseui"

import {
  TransformWrapper,
  TransformComponent,
  ReactZoomPanPinchRef,
} from "react-zoom-pan-pinch"
import { Button } from "baseui/button"
import { Switch } from "@base-ui-components/react/switch"

import switch_styles from "./MySwitch.module.css" // your custom styles

interface Fiber {
  width: number
  height: number
  x: number
  y: number
  fiber_id: string
  ratio: number
  type: string
  points: string[]
  colors: string[]
  is_error: boolean
}

const engine = new Styletron()

/**
 * A template for creating Streamlit components with React
 *
 *
 * @param {ComponentProps} props - The props object passed from Streamlit
 * @param {Object} props.args - Custom arguments passed from the Python side
 * @param {boolean} props.disabled - Whether the component is in a disabled state
 * @param {Object} props.theme - Streamlit theme object for consistent styling
 * @returns {ReactElement} The rendered component
 */
function FiberComponent(
  this: any,
  { args, disabled, width, theme }: ComponentProps
): ReactElement {
  // Extract custom arguments passed from Python
  let { image, elements, image_w, image_h } = args
  const [showOnlyPolylines, setShowOnlyPolylines] = useState(false)
  // Parse elements if it's a string (e.g., JSON)
  elements = elements.map((el: any): Fiber => {
    if (typeof el === "string") {
      return JSON.parse(el) as Fiber
    }
    return el as Fiber
  })

  // Component state
  const [isFocused, setIsFocused] = useState(false)
  const [hideErrors, setHideErrors] = useState(false)
  /**
   * Dynamic styling based on Streamlit theme and component state
   * This demonstrates how to use the Streamlit theme for consistent styling
   */

  const transformRef = useRef<ReactZoomPanPinchRef | null>(null)

  const style: React.CSSProperties = useMemo(() => {
    if (!theme) return {}

    // Use the theme object to style the button border
    // Access theme properties like primaryColor, backgroundColor, etc.
    const borderStyling = `1px solid ${isFocused ? theme.primaryColor : "gray"}`
    return { border: borderStyling, outline: borderStyling }
  }, [theme, isFocused])

  /**
   * Tell Streamlit the height of this component
   * This ensures the component fits properly in the Streamlit app
   */
  useEffect(() => {
    Streamlit.setFrameHeight()
  }, [style, theme])

  /**
   * Click handler for the button
   * Demonstrates how to update component state and send data back to Streamlit
   */

  const margin = (4 * Math.min(image_w, image_h)) / 1024
  const default_radius = Math.min(image_w, image_h) / 1024
  const handleToggle = () => {
    setShowOnlyPolylines((prev) => !prev)
  }

  const handleRecenter = () => {
    transformRef.current?.resetTransform()
  }
  const themeMode = theme?.base == "dark" ? LightTheme : DarkTheme
  return (
    <StyletronProvider value={engine}>
      <BaseProvider theme={themeMode}>
        <div
          style={{
            width: "100%",
          }}
          onKeyDown={(e) => {
            console.log(e.code)
            if (e.code === "KeyT" || e.key === " ") {
              handleToggle()
            }
          }}
        >
          <div
            style={
              {
                display: "flex",
                alignItems: "center",
                gap: "8px",
                "--color-gray-100":
                  (theme as any)?.colors?.backgroundTertiary ?? "#f0f0f0",
                "--color-gray-200":
                  (theme as any)?.colors?.borderOpaque ?? "#e0e0e0",
                "--color-gray-500":
                  (theme as any)?.colors?.borderSelected ?? "#b0b0b0",
                "--color-gray-700":
                  (theme as any)?.colors?.contentPrimary ?? "#333333",
                "--color-blue": (theme as any)?.colors?.accent ?? "#276ef1",
              } as React.CSSProperties
            }
          >
            <Button onClick={handleToggle}>Toggle View (T)</Button>

            <Button onClick={handleRecenter}>Recenter View</Button>
            <div style={{ display: "flex", alignItems: "center", gap: "4px" }}>
              <Switch.Root
                checked={hideErrors}
                onCheckedChange={setHideErrors}
                className={switch_styles.Switch}
              >
                <Switch.Thumb className={switch_styles.Thumb} />
              </Switch.Root>
              <span style={{ fontSize: "14px" }}>Hide errors</span>
            </div>
            <span>
              Found {elements.length} fibers (
              {elements.filter((el: Fiber) => el.is_error).length} with errors)
            </span>
          </div>
          <TransformWrapper
            ref={transformRef}
            disabled={disabled}
            minScale={0.75}
            maxScale={10}
            wheel={{
              disabled: false,
              smoothStep: 0.01,
              step: 0.5,
            }}
          >
            <TransformComponent>
              <svg
                style={{ backgroundColor: DarkTheme.colors.backgroundPrimary }}
                width={width}
                viewBox={`0 0 ${image_w} ${image_h}`}
                xmlns="http://www.w3.org/2000/svg"
              >
                <image
                  id="img"
                  width={image_w}
                  height={image_h}
                  href={image}
                  className={`image ${showOnlyPolylines ? " hidden" : ""}`}
                />

                {elements.map((el: Fiber, idx: number) => {
                  if (el.is_error && hideErrors) {
                    return null
                  }
                  return (
                    <g key={idx} className="rect-group">
                      <rect
                        x={el.x - margin}
                        y={el.y - margin}
                        width={el.width + margin * 2}
                        height={el.height + margin * 2}
                        fill="none"
                        stroke={el.is_error ? "red" : "blue"}
                        strokeWidth={default_radius}
                        className={`hover-target ${
                          showOnlyPolylines ? "hidden" : ""
                        }`}
                        rx={default_radius}
                      >
                        <title>
                          Fiber id: {el.fiber_id}, Ratio: {el.ratio.toFixed(2)}
                        </title>
                      </rect>

                      <g className="hover-paths">
                        {el.points.map((line: string, line_idx: number) => (
                          <polyline
                            className="fibers"
                            key={`${line_idx}_${idx}`}
                            points={line}
                            fill="none"
                            stroke={el.colors[line_idx]}
                            strokeWidth={default_radius}
                            opacity={showOnlyPolylines ? 1.0 : 0.0}
                          />
                        ))}
                      </g>
                    </g>
                  )
                })}
              </svg>
            </TransformComponent>
          </TransformWrapper>
        </div>
      </BaseProvider>
    </StyletronProvider>
  )
}

export default withStreamlitConnection(FiberComponent)
