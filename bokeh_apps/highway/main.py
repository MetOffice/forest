import bokeh.plotting


def main():
    app(bokeh.plotting.curdoc())


def app(document):
    figure = bokeh.plotting.figure()
    figure.circle([1, 2, 3], [1, 2, 3])
    document.add_root(figure)


if __name__ == '__main__' or __name__.startswith('bk'):
    main()
