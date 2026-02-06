import os

with open('file_list.txt', 'r') as f:
    files = f.readlines()

markdown_content = "# File list for verification\n\n"
markdown_content += "| № | Status | File path |\n"
markdown_content += "|---|:---:|---|\n"

for i, file_path in enumerate(files, 1):
    file_path = file_path.strip().lstrip('./')
    if not file_path:
        continue
    # Skipping the script itself and the temp file
    if file_path in ['file_list.txt', 'generate_markdown.py']:
        continue
    markdown_content += f"| {i} | [ ] | {file_path} |\n"

with open('file_audit/file_verification_list.md', 'w') as f:
    f.write(markdown_content)

print("Markdown file created at file_audit/file_verification_list.md")
