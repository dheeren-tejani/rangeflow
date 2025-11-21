Contributing to RangeFlowThank you for your interest in contributing to RangeFlow! We welcome contributions from the community to help make AI more robust and safe.Getting StartedFork the Repository: Click the "Fork" button on the top right of the GitHub page.Clone Your Fork:git clone [https://github.com/dheeren-tejani/rangeflow.git](https://github.com/dheeren-tejani/rangeflow.git)
cd rangeflow
Set Up a Virtual Environment:python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Dependencies:pip install -e .[dev,torch,gpu]
Development WorkflowCreate a Branch: Always work on a new branch for your changes.git checkout -b feature/my-new-feature
Write Code: Make your changes in the src/rangeflow directory.Run Tests: Ensure your changes don't break existing functionality.pytest tests/
# Or run the manual verification script
python verify.py
Add Documentation: If you add new features, please update the README.md or add docstrings.Pull Request ProcessCommit Your Changes:git add .
git commit -m "Add feature: Description of your changes"
Push to GitHub:git push origin feature/my-new-feature
Open a Pull Request: Go to the original repository on GitHub and click "New Pull Request". Describe your changes clearly.Code of ConductPlease be respectful and constructive in all interactions. We are here to learn and build together.LicenseBy contributing, you agree that your contributions