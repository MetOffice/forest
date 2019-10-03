"""Key press interactivity"""
import bokeh.models
from forest.observe import Observable
from forest.export import export


__all__ = []


KEY_PRESS = "KEY_PRESS"


def press(code):
    """Key press action creator"""
    return {
        "kind": KEY_PRESS,
        "payload": {
            "code": code
        }
    }


@export
class KeyPress(Observable):
    """Key press server-side observable

    .. note:: KeyPress.hidden_button must be added to the
              document to allow JS hack to initialise callbacks
    """
    def __init__(self):
        source = bokeh.models.ColumnDataSource({
            'keys': []
        })
        source.on_change('data', self.on_change)
        custom_js = bokeh.models.CustomJS(args=dict(source=source), code="""
            if (typeof keyPressOn === 'undefined') {
                document.onkeydown = function(e) {
                    source.data = {
                        'keys': [e.code]
                    }
                    source.change.emit()
                }
                // Global to prevent multiple onkeydown callbacks
                keyPressOn = true
            }
        """)
        self.hidden_button = bokeh.models.Button(
                css_classes=['keypress-hidden-btn'])
        self.hidden_button.js_on_click(custom_js)
        super().__init__()

    def on_change(self, attr, old, new):
        code = new[0]
        self.notify(press(code))
