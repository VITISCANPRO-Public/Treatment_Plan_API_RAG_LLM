from pathlib import Path
from typing import List, Dict, Any

import frontmatter


def load_markdown_fiches(knowledge_dir: str = "data/knowledge") -> List[Dict[str, Any]]:
    """
    Charge toutes les fiches .md dans data/knowledge et retourne
    une liste de dictionnaires contenant :
    - metadata (front matter),
    - content (texte brut).
    """

    base = Path(knowledge_dir)
    fiches: List[Dict[str, Any]] = []

    for md_path in base.glob("*.md"):
        post = frontmatter.load(md_path)
        fiches.append(
            {
                "path": str(md_path),
                "meta": dict(post.metadata),
                "content": post.content,
            }
        )

    return fiches


if __name__ == "__main__":
    fiches = load_markdown_fiches()
    print(f"Nombre de fiches trouvées : {len(fiches)}")
    for fiche in fiches:
        print("-", fiche["path"], "meta:", fiche["meta"])
