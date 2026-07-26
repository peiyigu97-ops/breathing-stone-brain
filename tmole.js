const { tunnelmole } = require('tunnelmole');
tunnelmole({ port: 7860 }).then(url => {
  console.log('URL:', url);
});
