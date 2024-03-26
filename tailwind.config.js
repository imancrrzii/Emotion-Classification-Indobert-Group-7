/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./templates/*"],
  theme: {
    extend: {
      colors:{
        "primary": "#010851",
        "secondary": "#FF5733", 
        "tartiary": "#999999", 
        "pink": "#EE9AE5",
        "warning": "#E7D60D",
        "light-yellow": "#f3eedf",
      },
      boxShadow: {
        '3xl': '0px 10px 50px 0px rgba(0, 0, 0, 0.15)'
      }
    },
  },
  plugins: [],
}