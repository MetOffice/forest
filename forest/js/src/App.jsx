import { createSignal, createEffect, Show } from "solid-js"

const DiagramIcon = props => {
    // Font Awesome chart-line icon
    return (
        <div onClick={props.onClick} class="py-4 px-2">
            <i class="fa fa-chart-line cursor-pointer hover:text-blue-600"></i>
        </div>
    )
}

const App = props => {
    // Diagrams open/close functionality
    const [ isOpen, setIsOpen ] = createSignal(false)

    const close = () => {
        setIsOpen(false)
    }
    const open = () => {
        setIsOpen(true)
    }
    createEffect(() => {
        // Side-effect from template implementation
        const el = document.getElementById("diagrams")
        const width = "400px"
        if (isOpen()) {
            el.style.width = width
        } else {
            el.style.width = "0"
        }
    })
    return (
        <Show when={isOpen()} fallback={<DiagramIcon onClick={open} />}>

                <div class="flex flex-row justify-between py-4 px-2 min-w-32">
                    <span class="font-family-helvetica font-semibold">Analysis</span>
                    <span class="close-icon"><i class="fas fa-window-close" onClick={close}></i></span>
                </div>
        </Show>
    )
}

export default App
