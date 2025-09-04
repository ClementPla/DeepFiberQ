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
} from "react"

import "./style.css"
import { Button } from "baseui/button"
interface Fiber {
  width: number
  height: number
  x: number
  y: number
  points: string[]
  colors: string[]
}

import { TransformWrapper, TransformComponent } from "react-zoom-pan-pinch"

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
  elements = elements.map((el: string): Fiber => {
    return JSON.parse(el) as Fiber
  })

  // Component state
  const [isFocused, setIsFocused] = useState(false)

  /**
   * Dynamic styling based on Streamlit theme and component state
   * This demonstrates how to use the Streamlit theme for consistent styling
   */
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
  const default_radius = (2 * Math.min(image_w, image_h)) / 1024

  const handleToggle = () => {
    setShowOnlyPolylines((prev) => !prev)
  }

  return (
    <div
      style={{
        backgroundColor: theme?.backgroundColor,
        width: "100%",
      }}
    >
      <Button
        style={style}
        onClick={handleToggle}
        tabIndex={0}
        autoFocus
        onKeyPress={(e) => {
          console.log(e.code)
          if (e.code === "KeyT" || e.key === " ") {
            handleToggle()
          }
        }}
      >
        Toggle View (T){" "}
      </Button>
      <TransformWrapper
        disabled={disabled}
        minScale={0.75}
        maxScale={5}
        wheel={{
          disabled: false,
          smoothStep: 0.005,
          step: 0.5,
        }}
      >
        <TransformComponent>
          <svg
            style={{ backgroundColor: theme?.backgroundColor }}
            width={width}
            viewBox={`0 0 ${image_w} ${image_h}`}
            xmlns="http://www.w3.org/2000/svg"
          >
            <defs>
              <filter id="gs">
                <feColorMatrix
                  in="SourceGraphic"
                  type="matrix"
                  values="1.1 0 0 0 -0.1
                                                              0 1.1 0 0 -0.1
                                                              0 0 1.1 0 -0.1
                                                              0 0 0 1 0"
                />
                <feColorMatrix type="saturate" values="0.0" />
              </filter>

              {elements.map((el: Fiber, idx: number) => (
                <clipPath id={`clip-${idx}`} key={`clip-${idx}`}>
                  <rect
                    x={el.x - margin - (margin * 2) / 3}
                    y={el.y - margin - (margin * 2) / 3}
                    width={el.width + 2 * (margin + (margin * 2) / 3) - 2}
                    height={el.height + 2 * (margin + (margin * 2) / 3) - 2}
                    rx={default_radius}
                  />
                </clipPath>
              ))}
            </defs>

            <image
              id="img"
              width={image_w}
              height={image_h}
              href={image}
              className={`image ${showOnlyPolylines ? " hidden" : ""}`}
            />

            {elements.map((el: Fiber, idx: number) => (
              <g key={idx} className="rect-group">
                <rect
                  x={el.x - margin}
                  y={el.y - margin}
                  width={el.width + margin * 2}
                  height={el.height + margin * 2}
                  fill="none"
                  stroke="blue"
                  strokeWidth={default_radius / 4}
                  className={`hover-target ${
                    showOnlyPolylines ? "hidden" : ""
                  }`}
                  rx={default_radius}
                />

                <use
                  href="#img"
                  filter="url(#gs)"
                  clipPath={`url(#clip-${idx})`}
                  className={`gray-patch ${showOnlyPolylines ? "hidden" : ""}`}
                />

                <g className="hover-paths">
                  {el.points.map((line: string, line_idx: number) => (
                    <polyline
                      className="fibers"
                      key={`${line_idx}_${idx}`}
                      points={line}
                      fill="none"
                      stroke={el.colors[line_idx]}
                      strokeWidth={default_radius}
                      opacity={showOnlyPolylines ? 1.0 : 0.15}
                    />
                  ))}
                </g>
              </g>
            ))}
          </svg>
        </TransformComponent>
      </TransformWrapper>
    </div>
  )
}

export default withStreamlitConnection(FiberComponent)
