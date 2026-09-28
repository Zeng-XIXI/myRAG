import {
  FrontendRenderer,
  FrontendRendererArgs,
} from "@streamlit/component-v2-lib";
import { StrictMode } from "react";
import { createRoot, Root } from "react-dom/client";

import CitationViewer, { CitationViewerData } from "./MyComponent";
import "./style.css";

const reactRoots: WeakMap<FrontendRendererArgs["parentElement"], Root> =
  new WeakMap();

const CitationViewerRoot: FrontendRenderer<Record<string, never>, CitationViewerData> =
  ({ data, parentElement }) => {
    const rootElement = parentElement.querySelector(".react-root");
    if (!rootElement) {
      throw new Error("Citation viewer root element not found");
    }

    let reactRoot = reactRoots.get(parentElement);
    if (!reactRoot) {
      reactRoot = createRoot(rootElement);
      reactRoots.set(parentElement, reactRoot);
    }

    reactRoot.render(
      <StrictMode>
        <CitationViewer answer={data.answer} sources={data.sources} />
      </StrictMode>,
    );

    return () => {
      const currentRoot = reactRoots.get(parentElement);
      if (currentRoot) {
        currentRoot.unmount();
        reactRoots.delete(parentElement);
      }
    };
  };

export default CitationViewerRoot;
