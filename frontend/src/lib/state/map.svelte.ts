function createMap() {
  let coordinates = $state("Barcelona");

  function setCoordinates(lon: number, lat: number) {
    coordinates = `${lon},${lat}`;
  }

  return {
    get coordinates() {
      return coordinates;
    },
    setCoordinates,
  };
}

export const map = createMap();
