const forest = require("./forest")


describe("helloWorld", () => {
    it("should return message", () => {
        expect(forest.helloWorld()).toEqual("Hello, World!")
    })
})
