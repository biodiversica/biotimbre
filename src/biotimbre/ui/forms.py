"""The item metadata form, shared by the add and edit flows.

One builder for both, so the two cannot drift apart — an edit form that accepted
a field the add form rejected would be a subtle and unpleasant bug.

Validation is delegated entirely to
:meth:`~biotimbre.domain.item.ItemDraft.validated`. The form's only job is to map
the resulting :class:`~biotimbre.domain.errors.ValidationError` back onto the
input that caused it, which is why the domain records a ``field`` name at all.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from nicegui import ui

from biotimbre.domain.errors import ValidationError
from biotimbre.domain.item import (
    DEFAULT_SPEED,
    FilteredBand,
    ItemDraft,
    LibraryItem,
    Location,
    SampleWindow,
)
from biotimbre.i18n import Translator
from biotimbre.ui.maps import location_picker
from biotimbre.ui.sections import foldable
from biotimbre.ui.suggestions import suggest_tags, suggest_text

#: Maps a ``ValidationError.field`` onto the input to highlight. Compound fields
#: report either the whole value or one bound, so both spellings are listed.
_FIELD_TO_INPUT: dict[str, str] = {
    "date": "date",
    "time": "time",
    "location": "latitude",
    "location.latitude": "latitude",
    "location.longitude": "longitude",
    "sample_window": "window_start",
    "sample_window.start": "window_start",
    "sample_window.end": "window_end",
    "filtered_band": "highpass",
    "filtered_band.low": "highpass",
    "filtered_band.high": "lowpass",
    "filter_type": "filter_type",
    "filter_order": "filter_order",
    "temperature": "temperature",
    "humidity": "humidity",
    "speed": "speed",
    "taxa": "taxa",
    "biotimbre_id": "biotimbre_id",
    "device": "device",
    "microphone": "microphone",
    "recorded_by": "recorded_by",
    "link_to_full_recording": "link",
    "notes": "notes",
}


#: Inputs shown above the fold; every other one lives in the optional panel.
_REQUIRED_INPUTS = frozenset({"date", "time", "latitude", "longitude"})


def _blank_to_none(value: object) -> object | None:
    """Treat an untouched input as absent rather than as an empty value."""
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value


@dataclass
class MetadataForm:
    """Inputs for one item's metadata, plus draft construction and error display.

    Build with :func:`metadata_form`.
    """

    t: Translator
    inputs: dict[str, ui.element]
    allow_new_biotimbre: bool = True
    #: The foldable holding everything but the required fields.
    optional_panel: ui.expansion | None = None

    def clear_errors(self) -> None:
        """Remove any error state left from a previous attempt."""
        for element in self.inputs.values():
            if hasattr(element, "props"):
                element.props(remove="error error-message")

    def show_error(self, exc: ValidationError) -> bool:
        """Highlight the input a validation error refers to.

        Args:
            exc: The error to display.

        Returns:
            True if an input was matched, False if the error is not
            field-specific and should be shown as a notification instead.
        """
        name = _FIELD_TO_INPUT.get(exc.field or "")
        element = self.inputs.get(name) if name else None
        if element is None:
            return False
        message = str(exc).split(": ", 1)[-1].replace('"', "'")
        element.props(f'error error-message="{message}"')
        if self.optional_panel is not None and name not in _REQUIRED_INPUTS:
            # A highlighted field nobody can see is no help.
            self.optional_panel.value = True
        return True

    def draft(self) -> ItemDraft:
        """Build a validated draft from the current input values.

        Returns:
            A validated :class:`~biotimbre.domain.item.ItemDraft`.

        Raises:
            ValidationError: If any value is unacceptable. Pass it to
                :meth:`show_error` to mark the offending input.
        """
        values = {name: element.value for name, element in self.inputs.items()}

        latitude = _blank_to_none(values["latitude"])
        longitude = _blank_to_none(values["longitude"])
        if latitude is None or longitude is None:
            raise ValidationError(
                self.t("common.required"), "location"
            )

        biotimbre_id = _blank_to_none(values["biotimbre_id"])
        if biotimbre_id is None and not self.allow_new_biotimbre:
            # An update never allocates a new biotimbre (repository.update_item)
            # — a blank field here would silently leave the item's biotimbre
            # unchanged, which looks like the edit did nothing. Reject it
            # instead: keep the existing ID, or type a different one to join.
            raise ValidationError(self.t("common.required"), "biotimbre_id")

        band = FilteredBand.parse(
            (_blank_to_none(values["highpass"]), _blank_to_none(values["lowpass"]))
        )

        return ItemDraft(
            date=values["date"] or "",
            time=values["time"] or "",
            location=Location.parse((latitude, longitude)),
            # Optional: the window locates the sample inside a full
            # recording, and not every sample comes from one.
            sample_window=SampleWindow.parse_optional(
                (
                    _blank_to_none(values["window_start"]),
                    _blank_to_none(values["window_end"]),
                )
            ),
            biotimbre_id=biotimbre_id,
            taxa=values["taxa"] or (),
            device=_blank_to_none(values["device"]),
            microphone=_blank_to_none(values["microphone"]),
            recorded_by=_blank_to_none(values["recorded_by"]),
            filtered_band=band,
            filter_type=_blank_to_none(values["filter_type"]),
            filter_order=_blank_to_none(values["filter_order"]),
            temperature=_blank_to_none(values["temperature"]),
            humidity=_blank_to_none(values["humidity"]),
            speed=values["speed"] if values["speed"] is not None else DEFAULT_SPEED,
            link_to_full_recording=_blank_to_none(values["link"]),
            notes=_blank_to_none(values["notes"]),
        ).validated()

    @property
    def speed(self) -> float:
        """Current speed value, for warning about a re-embed before saving."""
        value = self.inputs["speed"].value
        return float(value) if value is not None else DEFAULT_SPEED

    @property
    def label(self) -> str | None:
        """The biotimbre label field's current value, or ``None`` if blank."""
        value = self.inputs["biotimbre_label"].value
        if not isinstance(value, str):
            return None
        return value.strip() or None


def metadata_form(
    t: Translator,
    item: LibraryItem | None = None,
    *,
    allow_new_biotimbre: bool = True,
    biotimbre_label: str | None = None,
    map_picker: bool = False,
    suggest: Callable[[str, str], Awaitable[list[str]]] | None = None,
) -> MetadataForm:
    """Render the metadata inputs and return a handle to them.

    Args:
        t: Active translator.
        item: Existing item to pre-fill from, for the edit flow.
        allow_new_biotimbre: Whether an empty biotimbre ID is meaningful. On edit
            it is not — an update never allocates — so the field is required there.
        biotimbre_label: Current label of the item's biotimbre, to pre-fill the
            label field on edit. Left blank on add, since a freshly allocated
            biotimbre has none yet.
        map_picker: Whether to offer a map for setting the location by
            clicking. Off unless asked for, since the map is switchable.
        suggest: Looks up values already in the library, as
            ``suggest(field, typed)``, to offer back while taxa, device,
            microphone and recorded-by are typed. ``None`` offers nothing.

    Returns:
        The :class:`MetadataForm` handle.
    """
    draft = item.to_draft() if item is not None else None
    inputs: dict[str, ui.element] = {}

    def dense(element):
        return element.props("dense outlined").classes("w-full")

    ui.label(t("admin.mandatory_section")).classes(
        "text-xs uppercase tracking-wide text-muted mt-1"
    )
    with ui.grid(columns=2).classes("w-full gap-3"):
        inputs["date"] = dense(
            ui.input(t("field.date"), value=draft.date if draft else "")
            .props('mask="####-##-##" hint="YYYY-MM-DD"')
        )
        inputs["time"] = dense(
            ui.input(t("field.time"), value=draft.time if draft else "")
            .props('mask="##:##:##" hint="HH:MM:SS"')
        )
        inputs["latitude"] = dense(
            ui.number(
                t("field.latitude"),
                value=draft.location.latitude if draft else None,
                format="%.6f",
                step=0.000001,
            )
        )
        inputs["longitude"] = dense(
            ui.number(
                t("field.longitude"),
                value=draft.location.longitude if draft else None,
                format="%.6f",
                step=0.000001,
            )
        )
    if map_picker:
        location_picker(t, inputs["latitude"], inputs["longitude"])

    # Folds away, since most additions need few of these and the form is long;
    # the choice is remembered, and an error in a folded field unfolds it.
    optional_panel = foldable(
        "optional_metadata",
        t("admin.optional_section"),
        heading_class="text-xs uppercase tracking-wide text-muted",
    ).classes("mt-3")
    with optional_panel, ui.column().classes("w-full gap-3 pt-2"):
        with ui.grid(columns=2).classes("w-full gap-3"):
            inputs["biotimbre_id"] = dense(
                ui.input(
                    t("field.biotimbre_id"),
                    value=draft.biotimbre_id if draft else "",
                    placeholder="" if allow_new_biotimbre else None,
                ).tooltip(
                    t("field.biotimbre_id_help")
                    if allow_new_biotimbre
                    else t("field.biotimbre_id_edit_help")
                )
            )
            inputs["biotimbre_label"] = dense(
                ui.input(
                    t("field.biotimbre_label"), value=biotimbre_label or ""
                ).tooltip(t("field.biotimbre_label_help"))
            )
            inputs["taxa"] = dense(
                ui.select(
                    options=[],
                    value=list(draft.taxa) if draft else [],
                    label=t("field.taxa"),
                    multiple=True,
                    new_value_mode="add-unique",
                )
                .props("use-chips")
                .tooltip(t("field.taxa_help"))
            )
            inputs["device"] = dense(
                ui.input(t("field.device"), value=(draft.device if draft else "") or "")
                .tooltip(t("field.device_help"))
            )
            inputs["microphone"] = dense(
                ui.input(
                    t("field.microphone"),
                    value=(draft.microphone if draft else "") or "",
                ).tooltip(t("field.microphone_help"))
            )
            inputs["recorded_by"] = dense(
                ui.input(t("field.recorded_by"), value=(draft.recorded_by if draft else "") or "")
            )
            inputs["speed"] = dense(
                ui.number(
                    t("field.speed"),
                    value=draft.speed if draft else DEFAULT_SPEED,
                    step=0.05,
                    min=0.01,
                    max=100.0,
                ).tooltip(t("field.speed_help"))
            )
            inputs["highpass"] = dense(
                ui.number(
                    t("field.highpass"),
                    value=(draft.filtered_band.low if draft else None),
                    step=100,
                    suffix=t("common.hertz"),
                ).tooltip(t("field.band_help"))
            )
            inputs["lowpass"] = dense(
                ui.number(
                    t("field.lowpass"),
                    value=(draft.filtered_band.high if draft else None),
                    step=100,
                    suffix=t("common.hertz"),
                ).tooltip(t("field.band_help"))
            )
            inputs["filter_type"] = dense(
                ui.input(
                    t("field.filter_type"),
                    value=(draft.filter_type if draft else "") or "",
                ).tooltip(t("field.filter_type_help"))
            )
            inputs["filter_order"] = dense(
                ui.number(
                    t("field.filter_order"),
                    value=(draft.filter_order if draft else None),
                    step=1,
                    min=1,
                ).tooltip(t("field.filter_order_help"))
            )
            inputs["temperature"] = dense(
                ui.number(
                    t("field.temperature"),
                    value=(draft.temperature if draft else None),
                    step=0.1,
                    suffix="°C",
                )
            )
            inputs["humidity"] = dense(
                ui.number(
                    t("field.humidity"),
                    value=(draft.humidity if draft else None),
                    step=0.1,
                    min=0,
                    max=100,
                    suffix="%",
                )
            )
            inputs["link"] = dense(
                ui.input(
                    t("field.link"),
                    value=(draft.link_to_full_recording if draft else "") or "",
                )
            )
            inputs["window_start"] = dense(
                ui.number(
                    t("field.window_start"),
                    value=(
                        draft.sample_window.start
                        if draft and draft.sample_window
                        else None
                    ),
                    step=0.1,
                    suffix=t("common.seconds"),
                ).tooltip(t("field.window_help"))
            )
            inputs["window_end"] = dense(
                ui.number(
                    t("field.window_end"),
                    value=(
                        draft.sample_window.end
                        if draft and draft.sample_window
                        else None
                    ),
                    step=0.1,
                    suffix=t("common.seconds"),
                ).tooltip(t("field.window_help"))
            )

        inputs["notes"] = dense(
            ui.textarea(t("field.notes"), value=(draft.notes if draft else "") or "")
            .props("autogrow")
        )

    if suggest is not None:
        suggest_tags(inputs["taxa"], lambda text: suggest("taxa", text))
        for name in ("device", "microphone", "recorded_by"):
            suggest_text(
                inputs[name], lambda text, name=name: suggest(name, text)
            )

    return MetadataForm(
        t=t,
        inputs=inputs,
        allow_new_biotimbre=allow_new_biotimbre,
        optional_panel=optional_panel,
    )
