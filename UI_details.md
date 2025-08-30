The UI should be a single-page chat interface that dynamically transitions from an initial landing state to a full chat window. The layout must be divided into three distinct vertical sections.

#### **Technical and Structural Specifications:**

The homepage will be composed of three main layout sections using CSS Flexbox or Grid, with a responsive design for different screen sizes.

* **Main Content Area (60% width):** This is the central chat interface.
    * **Initial State:** Display a large, centered text input box. The box should have placeholder text that reads "**Ask anything**."
    * **File Upload:** A subtle "+" icon should be located below the "Ask anything" text, allowing users to upload files for processing.
    * **Model Selection:** A dropdown menu, labeled "**SELECT MODEL**," should be positioned in the bottom right corner of the input box.
    * **Dynamic Transition:** When a user begins typing in the input box, the entire component should smoothly transition into a full chat interface, similar to ChatGPT or Gemini. This involves the input box shrinking and moving to the bottom of the screen, and a chat history area appearing above it. The transition should be implemented using CSS animations and a JavaScript framework (e.g., React, Vue, or Angular) to manage the state change.

* **3D Avatar Canvas (30% width):** This section will be on the right side of the main content area.
    * **Interactive 3D Model:** The canvas will host a real-time, interactive 3D model. This model should respond to user queries with **facial animations, voice, and expressions**.
    * **Model/Avatar Selection:** A dropdown menu, labeled "**SELECT AVATAR**," will be located at the top of this section, allowing the user to change the 3D character.
    * **Technology Stack:** The 3D rendering should be handled using a library like **Three.js** or **Babylon.js** with **WebGL**. The avatar's voice and animations will be driven by an API that converts text responses into lip-sync data and emotional cues.

* **Sidebar Navigation (10% width):** This left-aligned vertical bar will contain icons for settings and other features.
    * **Collapsed State:** The sidebar will be in a contracted form, occupying 10% of the screen width.
    * **Expanded State:** The sidebar should expand to **25%-30% of the screen width** when a user clicks on an icon or hovers over the bar. This expansion should be a smooth animation.

**The overall design should be modern, clean, and minimalist, using a grayscale color palette as seen in the provided image.**

