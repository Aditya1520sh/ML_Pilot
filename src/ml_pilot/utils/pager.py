"""Paged display of long Rich/text content in the terminal.

Purpose:
    Avoid flooding the console when printing large EDA reports or leaderboards.

Interactions:
    - Used by EDA and Compare stages when verbose output is large.
"""

from __future__ import annotations

from rich.console import Console, RenderableType
from rich.pager import Pager


def page_renderable(console: Console, renderable: RenderableType, *, enabled: bool = True) -> None:
    """Print ``renderable``, optionally through the system pager.

    Args:
        console: Rich console instance.
        renderable: Object Rich can render.
        enabled: When False, prints without paging.
    """
    if not enabled:
        console.print(renderable)
        return
    with console.pager(styles=True):
        console.print(renderable)


class NullPager(Pager):
    """Pager that simply prints — useful in non-interactive environments."""

    def _pager(self, content: str) -> None:
        print(content)
