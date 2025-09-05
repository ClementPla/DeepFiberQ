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
import { Client as Styletron } from "styletron-engine-monolithic"
import { Provider as StyletronProvider } from "styletron-react"
import { LightTheme, BaseProvider, styled } from "baseui"
import { StatefulInput } from "baseui/input"

import { FileRow, FileUploader } from "baseui/file-uploader"
import "./style.css"

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
function LoaderComponent(
  this: any,
  { args, disabled, theme }: ComponentProps
): ReactElement {
  // Extract custom arguments passed from Python
  const [showOnlyPolylines, setShowOnlyPolylines] = useState(false)

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

  const [fileRows, setFileRows] = React.useState<FileRow[]>([])

  const engine = new Styletron()
  const Centered = styled("div", {
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    height: "100%",
  })

  return (
    <StyletronProvider value={engine}>
      <BaseProvider theme={LightTheme}>
        <FileUploader
          fileRows={fileRows}
          setFileRows={(newFileRows) => console.log(newFileRows)}
        />
      </BaseProvider>
    </StyletronProvider>
  )
}

export default withStreamlitConnection(LoaderComponent)
