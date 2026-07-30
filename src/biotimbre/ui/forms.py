"""The item metadata form, shared by the add and edit flows.

One builder for both, so the two cannot drift apart — an edit form that accepted
a field the add form rejected would be a subtle and unpleasant bug.

Validation is delegated entirely to
:meth:`~biotimbre.domain.item.ItemDraft.validated`. The form's only job is to map
the resulting :class:`~biotimbre.domain.errors.ValidationError` back onto the
input that caused it, which is why the domain records a ``field`` name at all.
"""

from __future__ import annotations

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
    "noise_reduction": "noise_reduction",
    "speed": "speed",
    "taxa": "taxa",
    "biotimbre_id": "biotimbre_id",
    "device": "device",
    "link_to_full_recording": "link",
    "notes": "notes",
}


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

        start = _blank_to_none(values["window_start"])
        end = _blank_to_none(values["window_end"])
        if start is None or end is None:
            raise ValidationError(self.t("common.required"), "sample_window")

        band = FilteredBand.parse(
            (_blank_to_none(values["highpass"]), _blank_to_none(values["lowpass"]))
        )

        return ItemDraft(
            date=values["date"] or "",
            time=values["time"] or "",
            location=Location.parse((latitude, longitude)),
            sample_window=SampleWindow.parse((start, end)),
            biotimbre_id=_blank_to_none(values["biotimbre_id"]),
            taxa=values["taxa"] or (),
            device=_blank_to_none(values["device"]),
            filtered_band=band,
            noise_reduction=bool(values["noise_reduction"]),
            speed=values["speed"] if values["speed"] is not None else DEFAULT_SPEED,
            link_to_full_recording=_blank_to_none(values["link"]),
            notes=_blank_to_none(values["notes"]),
        ).validated()

    @property
    def speed(self) -> float:
        """Current speed value, for warning about a re-embed before saving."""
        value = self.inputs["speed"].value
        return float(value) if value is not None else DEFAULT_SPEED


def metadata_form(
    t: Translator,
    item: LibraryItem | None = None,
    *,
    allow_new_biotimbre: bool = True,
) -> MetadataForm:
    """Render the metadata inputs and return a handle to them.

    Args:
        t: Active translator.
        item: Existing item to pre-fill from, for the edit flow.
        allow_new_biotimbre: Whether an empty biotimbre ID is meaningful. On edit
            it is not — an update never allocates — so the field is required there.

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
        inputs["window_start"] = dense(
            ui.number(
                t("field.window_start"),
                value=draft.sample_window.start if draft else None,
                step=0.1,
                suffix=t("common.seconds"),
            ).tooltip(t("field.window_help"))
        )
        inputs["window_end"] = dense(
            ui.number(
                t("field.window_end"),
                value=draft.sample_window.end if draft else None,
                step=0.1,
                suffix=t("common.seconds"),
            ).tooltip(t("field.window_help"))
        )

    ui.label(t("admin.optional_section")).classes(
        "text-xs uppercase tracking-wide text-muted mt-3"
    )
    with ui.grid(columns=2).classes("w-full gap-3"):
        inputs["biotimbre_id"] = dense(
            ui.input(
                t("field.biotimbre_id"),
                value=draft.biotimbre_id if draft else "",
                placeholder="" if allow_new_biotimbre else None,
            ).tooltip(t("field.biotimbre_id_help"))
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
        inputs["link"] = dense(
            ui.input(
                t("field.link"),
                value=(draft.link_to_full_recording if draft else "") or "",
            )
        )
        with ui.column().classes("justify-center"):
            inputs["noise_reduction"] = ui.switch(
                t("field.noise_reduction"),
                value=draft.noise_reduction if draft else False,
            )

    inputs["notes"] = dense(
        ui.textarea(t("field.notes"), value=(draft.notes if draft else "") or "")
        .props("autogrow")
    )

    return MetadataForm(t=t, inputs=inputs)
