class CaptureProcessor extends AudioWorkletProcessor {

  process(inputs) {

    const input = inputs[0]

    if (input && input[0]) {

      const channel = input[0]

      const copy = new Float32Array(channel.length)

      copy.set(channel)

      this.port.postMessage(copy)

    }

    return true
  }

}

registerProcessor('capture', CaptureProcessor)