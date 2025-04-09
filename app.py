# app.py (Temporary Diagnostic Code)
import streamlit as st
import sys
import platform
import importlib.metadata # Use this for Python 3.8+

st.set_page_config(layout="wide") # Use more space

st.write(f"## System Info")
st.write(f"Python Version: {sys.version}")
st.write(f"Platform: {platform.platform()}")

st.write(f"## Installed Packages (Attempting to List)")
try:
    installed_packages = {dist.metadata['Name'].lower(): dist.version for dist in importlib.metadata.distributions()}
    st.write(f"Found {len(installed_packages)} packages.")

    packages_to_check = [
        'streamlit', 'llama-index-core', 'llama-index-llms-huggingface',
        'llama-index-embeddings-huggingface', 'llama-index-readers-file',
        'sentence-transformers', 'transformers', 'huggingface-hub', 'torch',
        'python-dotenv', 'gdown', 'accelerate', 'safetensors' # Add others if relevant
    ]

    packages_found = {}
    for pkg in packages_to_check:
        pkg_lower = pkg.lower()
        if pkg_lower in installed_packages:
            packages_found[pkg] = installed_packages[pkg_lower]
        else:
             # Try common variations like replacing dashes
             pkg_alt = pkg_lower.replace('-', '_')
             if pkg_alt in installed_packages:
                 packages_found[pkg] = installed_packages[pkg_alt]
             else:
                packages_found[pkg] = "--- NOT FOUND ---"

    st.dataframe(packages_found, width=600)

except Exception as e_pkg:
    st.error(f"Could not list packages: {e_pkg}")


st.write(f"## Import Test")
try:
    st.write("Attempting: `import llama_index`")
    import llama_index
    st.success("`import llama_index` -- OK")

    st.write("Attempting: `from llama_index.llms.huggingface import HuggingFaceInferenceAPI`")
    # The line below is the one causing the error in your app
    from llama_index.llms.huggingface import HuggingFaceInferenceAPI
    st.success("`from llama_index.llms.huggingface import HuggingFaceInferenceAPI` -- OK")

    # Optional: Try other related imports if the above succeeds
    st.write("Attempting: `from llama_index.embeddings.huggingface import HuggingFaceEmbedding`")
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    st.success("`from llama_index.embeddings.huggingface import HuggingFaceEmbedding` -- OK")

except ImportError as e_imp:
    st.error(f"ImportError: {e_imp}")
    st.error(f"ImportError Details: Name='{e_imp.name}', Path='{e_imp.path}'")
    # Attempt to show traceback details if possible
    import traceback
    st.code(traceback.format_exc())

except Exception as e_gen:
     st.error(f"General Error during import: {e_gen}")
     import traceback
     st.code(traceback.format_exc())

st.write("--- Import test complete ---")
