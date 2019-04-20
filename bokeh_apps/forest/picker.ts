import {DatePicker, DatePickerView} from "models/widgets/date_picker"


export class DayPickerView extends DatePickerView {
    connect_signals(): void {
        super.connect_signals()
        this.connect(this.model.properties.value.change, () => {
            let date = new Date(this.model.value)
            this.inputEl.value = date.toDateString()
            this._picker.setDate(date)
        })
    }
}

export class DayPicker extends DatePicker {
    default_view = DayPickerView
    type = "DayPicker"
}
