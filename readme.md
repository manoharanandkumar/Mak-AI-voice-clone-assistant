# Mak AI

A Colab prototype for AI chat with browser microphone recording or audio upload, a voice-reference profile, and spoken responses.

## Run in Google Colab

1. Select **Runtime → Change runtime type → T4 GPU**.
2. Clone your repository:

   ```python
   !git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git
   %cd YOUR-REPO
   ```

3. Review the XTTS v2 model terms before using it. After you have reviewed and accepted them, install dependencies and start the app:

   ```python
   !pip install -r requirements.txt
   import os
   os.environ["COQUI_TOS_AGREED"] = "1"
   %run app.py
   ```

4. Open the temporary Gradio link printed by the app. It works only while the Colab session is running.

## Use

1. Check the consent box.
2. Record or upload a clear voice sample, then click **Save voice**.
3. Play the saved-reference preview to check that it sounds right.
4. Type a message or record/upload a question, then click **Ask Mak AI**.
5. Use **Delete voice** to remove the saved reference from the current session.

Use only your own voice or a voice you have explicit permission to use. Audio and model processing happen in the Colab runtime. Colab storage is temporary; the runtime reset removes session files. The app's share link is temporary, but anyone with access to it may be able to use the demo.

XTTS generates new speech using a reference recording; it does not reproduce the recording exactly. Check the model and dependency licenses before redistributing or using this project commercially.