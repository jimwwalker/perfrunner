function OnUpdate(doc, meta) {
    var fuzz = Math.floor(Math.random() * fuzz_factor);
    var fireAt = new Date(fixed_expiry + (fuzz * 1000));
    createTimer(timerCallback, fireAt, meta.id);
}

function timerCallback() {
    var request = {
        headers: {
            'Content-Type' : 'application/json'
        },
        body : {
            'key' : '01234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901'
        }
    };

    curl('POST', requestUrl, request);
}