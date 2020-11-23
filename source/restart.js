const path = require("path")
const { existsSync, writeFileSync } = require("fs")
const { execSync } = require('child_process')

console.log(execSync("hello").toString())

