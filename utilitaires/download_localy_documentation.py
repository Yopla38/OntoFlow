import os
import aiohttp
import asyncio

DOCUMENTS = [
    {"filepath": "https://gitlab.com/l_sim/bigdft-suite/-/raw/devel/bigdft-doc/school/System-Manipulation.ipynb", "project_name": "BigDFT", "version": "1.9"},
    {"filepath": "https://gitlab.com/l_sim/bigdft-suite/-/raw/devel/bigdft-doc/school/Introduction.ipynb", "project_name": "BigDFT", "version": "1.9"},
    {"filepath": "https://gitlab.com/l_sim/bigdft-suite/-/raw/devel/bigdft-doc/school/System-Generation.ipynb", "project_name": "BigDFT", "version": "1.9"},
    {"filepath": "https://gitlab.com/l_sim/bigdft-suite/-/raw/devel/bigdft-doc/school/QuickStart.ipynb", "project_name": "BigDFT", "version": "1.9"},
    {"filepath": "https://gitlab.com/l_sim/bigdft-suite/-/raw/devel/bigdft-doc/school/CMDStart.ipynb", "project_name": "BigDFT", "version": "1.9"},
    {"filepath": "https://gitlab.com/l_sim/bigdft-suite/-/raw/devel/bigdft-doc/school/implicit-solvation.ipynb", "project_name": "BigDFT", "version": "1.9"},
    {"filepath": "https://gitlab.com/l_sim/bigdft-suite/-/raw/devel/bigdft-doc/school/InitialRunsLinearScaling.ipynb", "project_name": "BigDFT", "version": "1.9"},
    {"filepath": "https://gitlab.com/l_sim/bigdft-suite/-/raw/devel/bigdft-doc/school/InitialRuns.ipynb", "project_name": "BigDFT", "version": "1.9"},

    {"filepath": "https://gitlab.com/l_sim/bigdft-suite/-/raw/devel/bigdft-doc/tutorials/IO.ipynb", "project_name": "BigDFT", "version": "1.9"},
    {"filepath": "https://gitlab.com/l_sim/bigdft-suite/-/raw/devel/bigdft-doc/tutorials/Rototranslations.ipynb", "project_name": "BigDFT", "version": "1.9"},
    {"filepath": "https://gitlab.com/l_sim/bigdft-suite/-/raw/devel/bigdft-doc/tutorials/N2-solution.ipynb", "project_name": "BigDFT", "version": "1.9"},
    {"filepath": "https://gitlab.com/l_sim/bigdft-suite/-/raw/devel/bigdft-doc/tutorials/Interoperability-Visualization.ipynb", "project_name": "BigDFT", "version": "1.9"},
    {"filepath": "https://gitlab.com/l_sim/bigdft-suite/-/raw/devel/bigdft-doc/tutorials/CH4.ipynb", "project_name": "BigDFT", "version": "1.9"},
    {"filepath": "https://gitlab.com/l_sim/bigdft-suite/-/raw/devel/bigdft-doc/tutorials/DoS-Manipulation.ipynb", "project_name": "BigDFT", "version": "1.9"},
    {"filepath": "https://gitlab.com/l_sim/bigdft-suite/-/raw/devel/bigdft-doc/tutorials/Systems.ipynb", "project_name": "BigDFT", "version": "1.9"},
    {"filepath": "https://gitlab.com/l_sim/bigdft-suite/-/raw/devel/bigdft-doc/tutorials/SolidState.ipynb", "project_name": "BigDFT", "version": "1.9"},
    {"filepath": "https://gitlab.com/l_sim/bigdft-suite/-/raw/devel/bigdft-doc/tutorials/CH4_aiida.ipynb", "project_name": "BigDFT", "version": "1.9"},
    {"filepath": "https://gitlab.com/l_sim/bigdft-suite/-/raw/devel/bigdft-doc/tutorials/Tight-Binding.ipynb", "project_name": "BigDFT", "version": "1.9"},
    {"filepath": "https://gitlab.com/l_sim/bigdft-suite/-/raw/devel/bigdft-doc/tutorials/N2.ipynb", "project_name": "BigDFT", "version": "1.9"},
    {"filepath": "https://gitlab.com/l_sim/bigdft-suite/-/raw/devel/bigdft-doc/tutorials/PDoS.ipynb", "project_name": "BigDFT", "version": "1.9"},
    {"filepath": "https://gitlab.com/l_sim/bigdft-suite/-/raw/devel/bigdft-doc/tutorials/Logfile-basics.ipynb", "project_name": "BigDFT", "version": "1.9"},
    {"filepath": "https://gitlab.com/l_sim/bigdft-suite/-/raw/devel/bigdft-doc/tutorials/Datasets.ipynb", "project_name": "BigDFT", "version": "1.9"},
    {"filepath": "https://gitlab.com/l_sim/bigdft-suite/-/raw/devel/bigdft-doc/tutorials/CalculatorsExamples.ipynb", "project_name": "BigDFT", "version": "1.9"},
    {"filepath": "https://gitlab.com/l_sim/bigdft-suite/-/raw/devel/bigdft-doc/tutorials/Interoperability-Simulation.ipynb", "project_name": "BigDFT", "version": "1.9"},

    {"filepath": "https://gitlab.com/l_sim/bigdft-suite/-/raw/devel/bigdft-doc/lessons/MachineLearning.ipynb", "project_name": "BigDFT", "version": "1.9"},
    {"filepath": "https://gitlab.com/l_sim/bigdft-suite/-/raw/devel/bigdft-doc/lessons/ComplexityReduction.ipynb", "project_name": "BigDFT", "version": "1.9"},
    {"filepath": "https://gitlab.com/l_sim/bigdft-suite/-/raw/devel/bigdft-doc/lessons/Gaussian.ipynb", "project_name": "BigDFT", "version": "1.9"},
    {"filepath": "https://gitlab.com/l_sim/bigdft-suite/-/raw/devel/bigdft-doc/lessons/GeometryOptimization.ipynb", "project_name": "BigDFT", "version": "1.9"},
    {"filepath": "https://gitlab.com/l_sim/bigdft-suite/-/raw/devel/bigdft-doc/lessons/MolecularDynamics.ipynb", "project_name": "BigDFT", "version": "1.9"},
]

OUTPUT_DIR = "downloaded_docs"

async def download_file(session, url, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    async with session.get(url) as response:
        response.raise_for_status()
        with open(save_path, "wb") as f:
            while True:
                chunk = await response.content.read(8192)
                if not chunk:
                    break
                f.write(chunk)
    print(f"✅ Saved {save_path}")


async def download_documents():
    async with aiohttp.ClientSession() as session:
        tasks = []
        for doc in DOCUMENTS:
            url = doc["filepath"]
            # Récupère le sous-répertoire (ex: tutorials, lessons, school)
            parts = url.split("/")
            if "school" in parts:
                subdir = "school"
            elif "tutorials" in parts:
                subdir = "tutorials"
            elif "lessons" in parts:
                subdir = "lessons"
            else:
                subdir = "misc"

            filename = parts[-1]  # ex: System-Manipulation.ipynb
            save_path = os.path.join(OUTPUT_DIR, subdir, filename)
            tasks.append(download_file(session, url, save_path))
        await asyncio.gather(*tasks)


asyncio.run(download_documents())
