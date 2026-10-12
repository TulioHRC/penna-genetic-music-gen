"""
Formulário interativo (TUI) para configurar uma execução do Penna.

Os campos são gerados a partir do próprio `argparse` de main.py, então toda opção nova
da CLI aparece aqui automaticamente:
- grupos do parser   -> seções do formulário
- store_true         -> Switch
- choices            -> Select
- nargs="+"          -> TextArea (um item por linha)
- int / float / str  -> Input já preenchido com o valor padrão

Os valores iniciais vêm do config.json (via defaults do parser); Ctrl+S grava o formulário
de volta nele, tornando os valores atuais o novo padrão da CLI e do formulário.
"""

import argparse
from typing import Any, Dict, List, Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Footer, Header, Input, Label, Select, Static, Switch, TextArea

from config import CONFIG_FILE, OPTIONAL_KEYS, load_defaults, save_defaults

HIDDEN_DESTS = {"help"}


def _form_actions(parser: argparse.ArgumentParser) -> Dict[str, List[argparse.Action]]:
    """Ações do parser agrupadas por seção, na ordem em que foram declaradas."""
    sections: Dict[str, List[argparse.Action]] = {}
    for group in parser._action_groups:
        actions = [a for a in group._group_actions if a.dest not in HIDDEN_DESTS]
        if actions:
            sections[group.title] = actions
    return sections


def _is_list(action: argparse.Action) -> bool:
    return action.nargs in ("+", "*")


def _is_flag(action: argparse.Action) -> bool:
    return isinstance(action, argparse._StoreTrueAction)


class ArgumentForm(App[Optional[argparse.Namespace]]):
    TITLE = "Penna 🎸🧬"
    SUB_TITLE = "Geração musical com Algoritmo Genético"

    CSS = """
    Screen { background: $surface; }
    #form { padding: 0 2; }
    .section {
        border: round $primary;
        border-title-color: $accent;
        border-title-style: bold;
        padding: 0 1;
        margin: 1 0 0 0;
        height: auto;
    }
    .field { height: auto; margin: 0 0 1 0; }
    .field-name { text-style: bold; }
    .field-help { color: $text-muted; }
    .field Input { width: 60; }
    .field TextArea { height: 4; }
    .field Switch { width: auto; }
    .field Select { width: 30; }
    #status { padding: 0 2; color: $text-muted; }
    """

    BINDINGS = [
        Binding("ctrl+r", "run", "Gerar músicas", priority=True),
        Binding("ctrl+s", "save_defaults", "Salvar como padrão", priority=True),
        Binding("ctrl+d", "reset_defaults", "Recarregar config.json", priority=True),
        Binding("escape", "quit_form", "Sair", priority=True),
    ]

    def __init__(self, parser: argparse.ArgumentParser):
        super().__init__()
        self.parser = parser
        self.sections = _form_actions(parser)
        self.actions_by_dest = {a.dest: a for actions in self.sections.values() for a in actions}

    # ---------- Montagem ----------

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="form"):
            for title, actions in self.sections.items():
                section = Vertical(classes="section")
                section.border_title = title
                with section:
                    for action in actions:
                        yield from self._compose_field(action)
        yield Static("Tab/Shift+Tab navegam entre campos · listas: um item por linha", id="status")
        yield Footer()

    def _compose_field(self, action: argparse.Action) -> ComposeResult:
        with Vertical(classes="field"):
            yield Label(action.option_strings[-1], classes="field-name")
            if action.help:
                yield Label(action.help, classes="field-help")
            yield self._build_widget(action, action.default)

    @staticmethod
    def _build_widget(action: argparse.Action, value: Any):
        widget_id = f"field-{action.dest}"
        if _is_flag(action):
            return Switch(value=bool(value), id=widget_id)
        if action.choices:
            options = [(str(choice), choice) for choice in action.choices]
            return Select(options, value=value, allow_blank=False, id=widget_id)
        if _is_list(action):
            return TextArea("\n".join(value or []), id=widget_id)
        input_type = {int: "integer", float: "number"}.get(action.type, "text")
        return Input("" if value is None else str(value), type=input_type, id=widget_id)

    # ---------- Leitura e validação ----------

    def _set_widget_value(self, action: argparse.Action, value: Any) -> None:
        widget = self.query_one(f"#field-{action.dest}")
        if _is_flag(action):
            widget.value = bool(value)
        elif action.choices:
            widget.value = value
        elif _is_list(action):
            widget.text = "\n".join(value or [])
        else:
            widget.value = "" if value is None else str(value)

    def _read_value(self, action: argparse.Action) -> Any:
        widget = self.query_one(f"#field-{action.dest}")
        if _is_flag(action) or action.choices:
            return widget.value
        if _is_list(action):
            return [line.strip() for line in widget.text.splitlines() if line.strip()]

        raw = widget.value.strip()
        if not raw:
            if action.dest in OPTIONAL_KEYS:
                return None
            raise ValueError(f"{action.option_strings[-1]} não pode ficar vazio")
        try:
            return action.type(raw) if action.type else raw
        except ValueError:
            raise ValueError(f"{action.option_strings[-1]}: valor inválido '{raw}'") from None

    # ---------- Ações ----------

    def _read_form(self) -> Optional[Dict[str, Any]]:
        """Lê todos os campos; em caso de erro avisa, foca o campo e retorna None."""
        values = {}
        for dest, action in self.actions_by_dest.items():
            try:
                values[dest] = self._read_value(action)
            except ValueError as error:
                self.notify(str(error), severity="error", title="Campo inválido")
                self.query_one(f"#field-{dest}").focus()
                return None
        return values

    def action_run(self) -> None:
        values = self._read_form()
        if values is None:
            return
        if not (values["youtube"] or values["audio"] or values["references"]):
            self.notify("Informe ao menos um link do YouTube, áudio ou MIDI de referência.",
                        severity="warning", title="Sem referências")
            self.query_one("#field-youtube").focus()
            return
        self.exit(argparse.Namespace(**values))

    def action_save_defaults(self) -> None:
        values = self._read_form()
        if values is None:
            return
        save_defaults(values)
        self.notify(f"Salvo em {CONFIG_FILE.name}", title="Padrões atualizados")

    def action_reset_defaults(self) -> None:
        saved = load_defaults()
        for dest, action in self.actions_by_dest.items():
            self._set_widget_value(action, saved[dest])
        self.notify(f"Valores recarregados de {CONFIG_FILE.name}.")

    def action_quit_form(self) -> None:
        self.exit(None)


def edit_arguments(parser: argparse.ArgumentParser) -> Optional[argparse.Namespace]:
    """Abre o formulário e devolve os argumentos preenchidos (ou None se o usuário saiu)."""
    return ArgumentForm(parser).run()


if __name__ == "__main__":
    from main import build_parser, run

    args = edit_arguments(build_parser())
    if args is not None:
        run(args)
