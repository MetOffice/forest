


class Artist(object):
    def __init__(self, viewers, renderers):
        self.viewers = viewers
        self.renderers = renderers
        self.visible_state = None
        self.state = None

    def on_visible(self, visible_state):
        if self.visible_state is not None:
            # Hide deselected states
            lost_items = (
                    set(self.flatten(self.visible_state)) -
                    set(self.flatten(visible_state)))
            for key, i, _ in lost_items:
                self.renderers[key][i].visible = False

        # Sync visible states with menu choices
        states = set(self.flatten(visible_state))
        hidden = [(i, j) for i, j, v in states if not v]
        visible = [(i, j) for i, j, v in states if v]
        for i, j in hidden:
            self.renderers[i][j].visible = False
        for i, j in visible:
            self.renderers[i][j].visible = True

        self.visible_state = dict(visible_state)
        self.render()

    @staticmethod
    def flatten(state):
        items = []
        for key, flags in state.items():
            items += [(key, i, f) for i, f in enumerate(flags)]
        return items

    def on_state(self, state):
        # print("Artist: {}".format(state))
        self.state = state
        self.render()

    def render(self):
        if self.visible_state is None:
            return
        if self.state is None:
            return
        for name in self.visible_state:
            viewer = self.viewers[name]
            viewer.render(self.state)
