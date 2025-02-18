// ffi.ts

// Determine library extension based on
// your OS.
let libSuffix = "";
switch (Deno.build.os) {
  case "windows":
    libSuffix = "dll";
    break;
  case "darwin":
    libSuffix = "dylib";
    break;
  case "linux":
    libSuffix = "so";
    break;
}

const libName = `./add.${libSuffix}`;
// Open library and define exported symbols
const dylib = Deno.dlopen(libName, {
  "add": { parameters: ["isize", "isize"], result: "isize" },
});

// Call the symbol `add`
const result = dylib.symbols.add(35, 34); // 69

console.log(`Result from external addition of 35 and 34: ${result}`);
